package com.danila.uploadservice.model;

public class FileUploadMessage {
    private String filePath;
    private  String fileName;
    private long uploadTime;

    @Override
    public String toString() {
        return "FileUploadMessage{" +
                "filePath='" + filePath + '\'' +
                ", fileName='" + fileName + '\'' +
                ", uploadTime=" + uploadTime +
                '}';
    }

    public String getFilePath() {
        return filePath;
    }

    public void setFilePath(String filePath) {
        this.filePath = filePath;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public long getUploadTime() {
        return uploadTime;
    }

    public void setUploadTime(long uploadTime) {
        this.uploadTime = uploadTime;
    }
}
