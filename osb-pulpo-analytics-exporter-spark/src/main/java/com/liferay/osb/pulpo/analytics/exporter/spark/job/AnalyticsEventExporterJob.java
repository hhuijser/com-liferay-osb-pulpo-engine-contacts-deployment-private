/**
 * Copyright (c) 2000-present Liferay, Inc. All rights reserved.
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 2.1 of the License, or (at your option)
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 */

package com.liferay.osb.pulpo.analytics.exporter.spark.job;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.collect_list;
import static org.apache.spark.sql.functions.spark_partition_id;
import static org.apache.spark.sql.functions.struct;
import static org.apache.spark.sql.functions.to_json;

import com.liferay.lcs.messaging.MessageBusMessage;
import com.liferay.osb.lcs.messaging.spring.client.MessageBusClient;
import com.liferay.osb.lcs.registry.client.LCSRegistryClient;
import com.liferay.osb.lcs.spark.core.job.BaseLCSSparkJob;

import com.netflix.appinfo.InstanceInfo;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.RelationalGroupedDataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

/**
 * @author Riccardo Ferrari
 */
@Component
public class AnalyticsEventExporterJob extends BaseLCSSparkJob {

	public AnalyticsEventExporterJob(
		@Value("${analytics.cassandra.read.keyspace}")
			String analyticsCassandraReadKeyspace,
		@Value("${analytics.cassandra.read.table}")
			String analyticsCassandraReadTable,
		@Value("${analytics.export.destination.name}")
			String analyticsExportDestinationName,
		@Value("${applicationId}") String applicationId,
		JavaSparkContext javaSparkContext, LCSRegistryClient lcsRegistryClient,
		SparkSession sparkSession) {

		super(applicationId, javaSparkContext, sparkSession);

		_analyticsCassandraReadKeyspace = analyticsCassandraReadKeyspace;
		_analyticsCassandraReadTable = analyticsCassandraReadTable;
		_analyticsExportDestinationName = analyticsExportDestinationName;
		/**
		 * This class is instantiated and ran on the Driver. However following
		 * Spark transformations and actions are executed on the Executor.
		 *
		 * LCSRegistryClient is not serializable thus we mark it as transient
		 *
		 * On the Executors will not be available (nor we need it)
		 */
		_lcsRegistryClient = lcsRegistryClient;
	}

	@Override
	protected void doRun(String... args) {
		LocalDateTime now = LocalDateTime.now(ZoneId.of("Z"));

		LocalDateTime endLocalDateTime = now.truncatedTo(ChronoUnit.HOURS);

		LocalDateTime startLocalDateTime = endLocalDateTime.minusHours(3);

		if (_log.isDebugEnabled()) {
			_log.debug(
				"Start Date {}, End date {}, Timezone: {}", startLocalDateTime,
				endLocalDateTime, System.getProperty("user.timezone"));
		}

		/**
		 * At each job iteration we use the Registry client to fetch the
		 * target instance.
		 *
		 * This code is executed on the Driver
		 */
		InstanceInfo instanceInfo = _lcsRegistryClient.getInstanceInfo(
			_analyticsExportDestinationName);

		String instanceInfoHomePageUrl = instanceInfo.getHomePageUrl();

		/**
		 * This call open a reference to the Cassandra analytics keyspace.
		 * No action is performed until a Spark Action is called.
		 */
		Dataset<Row> analyticsEventDataset = readDataset(
			_analyticsCassandraReadKeyspace, _analyticsCassandraReadTable);

		String filter = getFilter(startLocalDateTime, endLocalDateTime);

		/**
		 * We tell cassandra to fetch data from the last 3 hours. The key part
		 * in the getFilter above is that we add partition keys so that most
		 * of the filtering happens at Cassandra side
		 */
		analyticsEventDataset = analyticsEventDataset.where(filter);

		/**
		 * We convert the dataset into a single column in JSON format
		 */
		analyticsEventDataset = analyticsEventDataset.select(
			to_json(struct(col("*"))).alias("chunk"));

		/**
		 * We repartition the data. Spark will allocate '200' task to process
		 * the whole dataset.
		 *
		 * WARNING This number is a wild guess, and most likely wrong. To make
		 * sure we are always sending a reasonable amount of data we should
		 * - count the number of records CT
		 * - P = CT / <number of records per chunk>
		 *     where P is the partition number
		 *
		 */
		analyticsEventDataset = analyticsEventDataset.repartition(200);

		/**
		 * We add an identifier per data partition. Will be used in the next
		 * aggregation
		 */
		analyticsEventDataset = analyticsEventDataset.withColumn(
			"pid", spark_partition_id());

		RelationalGroupedDataset analyticsEventDatasetByPid =
			analyticsEventDataset.groupBy("pid");

		/**
		 * We group by 'pid' or partition ID then we collect all the items into
		 * a single wide-row
		 */
		analyticsEventDataset = analyticsEventDatasetByPid.agg(
			collect_list("chunk").cast("string"));

		/**
		 * Each row (one per partition P) is applied with the sendAnalyticsEvent
		 * method
		 */
		analyticsEventDataset.foreach(
			row -> {
				sendAnalyticsEvent(instanceInfoHomePageUrl, row);
			});
	}

	protected String getFilter(
		LocalDateTime startLocalDateTime, LocalDateTime endLocalDateTime) {

		List<String> partitionKeys = getPartitionKeys(
			startLocalDateTime, endLocalDateTime);

		StringBuilder sb = new StringBuilder(partitionKeys.size() * 2 + 7);

		sb.append("partitionkey IN ('");

		for (int i = 0; i < partitionKeys.size() - 1; i++) {
			sb.append(partitionKeys.get(i));
			sb.append("', '");
		}

		DateTimeFormatter dateTimeFormatter = DateTimeFormatter.ofPattern(
			"yyyy-MM-dd HH:mm:ss+0000");

		sb.append(partitionKeys.get(partitionKeys.size() - 1));
		sb.append("') ");
		sb.append("AND createdate >= '");
		sb.append(startLocalDateTime.format(dateTimeFormatter));
		sb.append("' AND createdate < '");
		sb.append(endLocalDateTime.format(dateTimeFormatter));
		sb.append("'");

		if (_log.isDebugEnabled()) {
			_log.debug("Filter: " + sb.toString());
		}

		return sb.toString();
	}

	protected String getPartitionKey(LocalDateTime localDateTime) {
		DateTimeFormatter dateTimeFormatter = DateTimeFormatter.ofPattern(
			"yyyyMMddHH00", Locale.UK);

		return localDateTime.format(dateTimeFormatter);
	}

	protected List<String> getPartitionKeys(
		LocalDateTime startLocalDateTime, LocalDateTime endLocalDateTime) {

		List<String> partitionKeys = new ArrayList<>();

		String endPartitionKey = getPartitionKey(endLocalDateTime);
		String startPartitionKey = getPartitionKey(startLocalDateTime);

		if (endPartitionKey.equals(startPartitionKey)) {
			partitionKeys.add(startPartitionKey);
		}
		else {
			String partitionKey = endPartitionKey;
			LocalDateTime partitionLocalDateTime = endLocalDateTime;

			while (!startPartitionKey.equals(partitionKey)) {
				partitionKeys.add(partitionKey);

				partitionLocalDateTime = partitionLocalDateTime.minusMinutes(
					60);

				partitionKey = getPartitionKey(partitionLocalDateTime);
			}

			partitionKeys.add(startPartitionKey);
		}

		return partitionKeys;
	}

	/**
	 * We extract the payload ans wrap it into a MessageBusMessage. This code is
	 * execute on the Executor and is serialized from the Driver. For this
	 * reason we have to instantiate a MessageBusClient here.
	 * @param instanceInfoHomePageUrl
	 * @param row
	 */
	protected void sendAnalyticsEvent(String instanceInfoHomePageUrl, Row row) {
		String payload = row.getString(1);

		if (_log.isDebugEnabled()) {
			_log.debug(payload);
		}

		MessageBusMessage messageBusMessage = new MessageBusMessage();

		messageBusMessage.setDestinationName(_analyticsExportDestinationName);
		messageBusMessage.setPayload(payload);

		MessageBusClient messageBusClient = new MessageBusClient(
			new RestTemplate());

		messageBusClient.send(messageBusMessage, instanceInfoHomePageUrl);
	}

	private static final Logger _log = LoggerFactory.getLogger(
		AnalyticsEventExporterJob.class);

	private final String _analyticsCassandraReadKeyspace;
	private final String _analyticsCassandraReadTable;
	private final String _analyticsExportDestinationName;
	private final transient LCSRegistryClient _lcsRegistryClient;

}