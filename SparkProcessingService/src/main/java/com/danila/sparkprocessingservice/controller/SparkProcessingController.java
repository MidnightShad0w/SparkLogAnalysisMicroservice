package com.danila.sparkprocessingservice.controller;

import com.danila.sparkprocessingservice.model.FileUploadMessage;
import com.danila.sparkprocessingservice.service.SparkModelService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/spark/api")
public class SparkProcessingController {

    private final SparkModelService sparkModelService;

    @Autowired
    public SparkProcessingController(SparkModelService sparkModelService) {
        this.sparkModelService = sparkModelService;
    }

    @PostMapping("/process")
    public ResponseEntity<String> processCsvFile(@RequestBody FileUploadMessage message) {
        try {
            String csvPath = message.getFilePath();
            String modelPath = "s3a://bucket-spark/model";
            String resultPath = "s3a://bucket-spark/result-anomaly-logs";

            sparkModelService.processFile(csvPath, modelPath, resultPath);

            return ResponseEntity.ok("Модель успешно применена, результат сохранен в " + resultPath);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Ошибка при обработке файла: " + e.getMessage());
        }
    }
}

