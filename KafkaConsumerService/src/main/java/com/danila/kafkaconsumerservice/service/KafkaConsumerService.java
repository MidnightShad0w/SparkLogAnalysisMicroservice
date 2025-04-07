package com.danila.kafkaconsumerservice.service;

import com.amazonaws.ClientConfiguration;
import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.client.builder.AwsClientBuilder;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.danila.kafkaconsumerservice.model.FileUploadMessage;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.io.File;

@Service
public class KafkaConsumerService {

    @Value("${target.service.url}")
    private String targetServiceUrl;

    @Value("${minio.bucket}")
    private String bucketName;

    @Value("${minio.endpoint}")
    private String minioEndpoint;

    @Value("${minio.accessKey}")
    private String minioAccessKey;

    @Value("${minio.secretKey}")
    private String minioSecretKey;

    private AmazonS3 s3Client;

    private final RestTemplate restTemplate;

    public KafkaConsumerService() {
        this.restTemplate = new RestTemplate();
    }

    @PostConstruct
    public void initS3Client() {
        BasicAWSCredentials credentials = new BasicAWSCredentials(minioAccessKey, minioSecretKey);
        ClientConfiguration clientConfig = new ClientConfiguration();
        clientConfig.setSignerOverride("AWSS3V4SignerType");

        s3Client = AmazonS3ClientBuilder.standard()
                .withEndpointConfiguration(new AwsClientBuilder.EndpointConfiguration(minioEndpoint, "us-east-1"))
                .withPathStyleAccessEnabled(true)
                .withClientConfiguration(clientConfig)
                .withCredentials(new AWSStaticCredentialsProvider(credentials))
                .build();
    }

    @KafkaListener(topics = "logs.upload", groupId = "upload-group")
    public void listen(FileUploadMessage message) {
        System.out.println("Получено сообщение из Kafka: " + message);
        try {
            File localFile = new File(message.getFilePath());
            String key = "uploads/" + localFile.getName();
            s3Client.putObject(bucketName, key, localFile);
            System.out.println("Файл загружен в MinIO: " + key);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
