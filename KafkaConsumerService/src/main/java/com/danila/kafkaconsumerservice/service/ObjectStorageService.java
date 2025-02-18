package com.danila.kafkaconsumerservice.service;

import com.amazonaws.ClientConfiguration;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.client.builder.AwsClientBuilder;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.File;

@Service
public class ObjectStorageService {

    private final AmazonS3 s3Client;
    private final String bucketName;

    public ObjectStorageService(
            @Value("${cloudru.tenant-id}") String tenantId,
            @Value("${cloudru.key-id}") String keyId,
            @Value("${cloudru.key-secret}") String keySecret,
            @Value("${cloudru.endpoint}") String endpoint,
            @Value("${cloudru.region}") String region,
            @Value("${cloudru.bucket-name}") String bucketName) {

        this.bucketName = bucketName;
        String accessKey = tenantId + ":" + keyId;
        BasicAWSCredentials credentials = new BasicAWSCredentials(accessKey, keySecret);
        ClientConfiguration clientConfig = new ClientConfiguration();
        clientConfig.setSignerOverride("AWSS3V4SignerType");

        this.s3Client = AmazonS3ClientBuilder.standard()
                .withEndpointConfiguration(new AwsClientBuilder.EndpointConfiguration(endpoint, region))
                .withPathStyleAccessEnabled(true)
                .withClientConfiguration(clientConfig)
                .withCredentials(new AWSStaticCredentialsProvider(credentials))
                .build();
    }

    public void uploadFile(String key, File file) {
        s3Client.putObject(bucketName, key, file);
    }
}
