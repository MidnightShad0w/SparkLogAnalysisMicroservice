package com.danila.uploadservice.service;

import com.danila.uploadservice.model.FileUploadMessage;
import lombok.extern.java.Log;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

@Service
public class KafkaProducerService {
    private final KafkaTemplate<String, FileUploadMessage> kafkaTemplate;
    private final String TOPIC = "logs.upload";

    @Autowired
    public KafkaProducerService(KafkaTemplate<String, FileUploadMessage> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }

    public void sendFileUploadMessage(FileUploadMessage message) {
        System.out.println("Файл попал в кафка сервис - " + message.toString());
        kafkaTemplate.send(TOPIC, message);
    }
}
