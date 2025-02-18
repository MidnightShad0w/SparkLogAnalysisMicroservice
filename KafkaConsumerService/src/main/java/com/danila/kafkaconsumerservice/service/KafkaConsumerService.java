package com.danila.kafkaconsumerservice.service;

import com.danila.kafkaconsumerservice.model.FileUploadMessage;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;

import java.io.File;

@Service
public class KafkaConsumerService {
    private final ObjectStorageService objectStorageService;

    @Autowired
    public KafkaConsumerService(ObjectStorageService objectStorageService) {
        this.objectStorageService = objectStorageService;
    }

    @KafkaListener(topics = "logs.upload", groupId = "upload-group")
    public void listen(FileUploadMessage message) {
        System.out.println("Получено сообщение из Kafka: " + message);

        String localFilePath = message.getFilePath();
        File fileToUpload = new File(localFilePath);

        if (!fileToUpload.exists()) {
            System.err.println("Файл не найден: " + localFilePath);
            return;
        }

        String key = "uploads/" + fileToUpload.getName();

        try {
            objectStorageService.uploadFile(key, fileToUpload);
            System.out.println("Файл успешно загружен в Object Storage с ключом: " + key);
        } catch (Exception e) {
            System.err.println("Ошибка загрузки файла в Object Storage: " + e.getMessage());
        }
    }
}
