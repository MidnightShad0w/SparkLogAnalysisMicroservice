package com.danila.kafkaconsumerservice.service;

import com.danila.kafkaconsumerservice.model.FileUploadMessage;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;

@Service
public class KafkaConsumerService {

    @Value("${target.service.url}")
    private String targetServiceUrl;

    private final RestTemplate restTemplate;

    public KafkaConsumerService() {
        this.restTemplate = new RestTemplate();
    }

    @KafkaListener(topics = "logs.upload", groupId = "upload-group")
    public void listen(FileUploadMessage message) {
        System.out.println("Получено сообщение из Kafka: " + message);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<FileUploadMessage> request = new HttpEntity<>(message, headers);

        try {
            ResponseEntity<String> response = restTemplate.postForEntity(targetServiceUrl, request, String.class);
            System.out.println("POST-запрос отправлен, получен ответ: " + response.getStatusCode());
        } catch (Exception e) {
            System.err.println("Ошибка при отправке POST-запроса: " + e.getMessage());
        }
    }
}
