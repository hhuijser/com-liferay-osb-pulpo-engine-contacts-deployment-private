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

package com.liferay.osb.pulpo.analytics.exporter.spark;

import com.liferay.osb.lcs.spark.core.BaseScheduledApplication;

import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cloud.client.discovery.EnableDiscoveryClient;
import org.springframework.context.annotation.Configuration;

/**
 * @author Riccardo Ferrari
 */
@Configuration
@EnableAutoConfiguration
@EnableConfigurationProperties
@EnableDiscoveryClient
public class AnalyticsExporterSparkApplication
	extends BaseScheduledApplication {

	public static void main(String[] args) {
		run(AnalyticsExporterSparkApplication.class, args);
	}

}