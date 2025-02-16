package com.danila.uploadservice.controller;

import com.danila.uploadservice.model.FileUploadMessage;
import com.danila.uploadservice.model.FileUploadResponse;
import com.danila.uploadservice.service.FileStorageService;
import com.danila.uploadservice.service.KafkaProducerService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/upload")
public class FileUploadController {

    private final FileStorageService fileStorageService;
    private final KafkaProducerService kafkaProducerService;

    @Autowired
    public FileUploadController(FileStorageService fileStorageService, KafkaProducerService kafkaProducerService) {
        this.fileStorageService = fileStorageService;
        this.kafkaProducerService = kafkaProducerService;
    }

    //curl -X POST -F file=@"Z:\Диплом\SparkLogAnalysisMicroservice\SparkMLModel\data\partlogdata.csv" http://localhost:8080/upload
    @PostMapping
    public ResponseEntity<FileUploadResponse> uploadFile(@RequestParam("file") MultipartFile file) {
        String filePath = fileStorageService.saveFile(file);

        FileUploadMessage message = new FileUploadMessage();
        message.setFilePath(filePath);
        message.setFileName(file.getOriginalFilename());
        message.setUploadTime(System.currentTimeMillis());

        kafkaProducerService.sendFileUploadMessage(message);

        FileUploadResponse response = new FileUploadResponse();
        response.setMessage("File uploaded successfully");
        response.setFilePath(filePath);
        return ResponseEntity.ok(response);
    }
}
