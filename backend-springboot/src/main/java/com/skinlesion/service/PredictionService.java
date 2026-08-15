package com.skinlesion.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.skinlesion.client.MLServiceClient;
import com.skinlesion.dto.PredictionHistoryResponse;
import com.skinlesion.dto.PredictionResponse;
import com.skinlesion.entity.Prediction;
import com.skinlesion.exception.ResourceNotFoundException;
import com.skinlesion.repository.PredictionRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class PredictionService {

    private final MLServiceClient mlServiceClient;
    private final PredictionRepository predictionRepository;
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${file.upload-dir:uploads}")
    private String uploadDir;

    @SuppressWarnings("unchecked")
    public PredictionResponse predictLesion(MultipartFile file, boolean explainable) {
        validateFile(file);

        // 1. Store uploaded file locally
        String savedPath = saveFileLocally(file);

        // 2. Delegate ML inference to Python FastAPI ML microservice
        Map<String, Object> mlResult = mlServiceClient.predict(file, explainable);

        // 3. Extract ML results
        String prediction = (String) mlResult.get("prediction");
        String predictionDisplayName = (String) mlResult.get("prediction_display_name");
        String category = (String) mlResult.get("category");
        String riskLevel = (String) mlResult.get("risk_level");
        Double confidence = ((Number) mlResult.get("confidence")).doubleValue();
        Boolean uncertain = (Boolean) mlResult.get("uncertain");
        String uncertaintyMessage = (String) mlResult.get("uncertainty_message");
        String modelName = (String) mlResult.get("model_name");
        List<Map<String, Object>> topPredictions = (List<Map<String, Object>>) mlResult.get("top_predictions");
        Map<String, Double> probabilities = (Map<String, Double>) mlResult.get("probabilities");
        Map<String, Object> diseaseInfo = (Map<String, Object>) mlResult.get("disease_info");
        Boolean explanationAvailable = (Boolean) mlResult.get("explanation_available");
        String gradcamBase64 = (String) mlResult.get("gradcam_base64");

        // 4. Serialize top predictions to JSON string for DB storage
        String topJson = "";
        try {
            topJson = objectMapper.writeValueAsString(topPredictions);
        } catch (Exception ignored) {}

        // 5. Persist Prediction record into MySQL database
        Prediction entity = Prediction.builder()
                .imagePath(savedPath)
                .predictedClass(prediction)
                .predictionDisplayName(predictionDisplayName != null ? predictionDisplayName : prediction)
                .category(category)
                .riskLevel(riskLevel)
                .confidence(confidence)
                .modelName(modelName)
                .topPredictionsJson(topJson)
                .gradcamBase64(gradcamBase64)
                .build();

        Prediction savedEntity = predictionRepository.save(entity);

        // 6. Build clean DTO response
        return PredictionResponse.builder()
                .predictionId(savedEntity.getId())
                .prediction(prediction)
                .predictionDisplayName(predictionDisplayName)
                .category(category)
                .riskLevel(riskLevel)
                .confidence(confidence)
                .uncertain(uncertain)
                .uncertaintyMessage(uncertaintyMessage)
                .modelName(modelName)
                .topPredictions(topPredictions)
                .probabilities(probabilities)
                .diseaseInfo(diseaseInfo)
                .explanationAvailable(explanationAvailable)
                .gradcamBase64(gradcamBase64)
                .timestamp(savedEntity.getCreatedAt())
                .build();
    }

    public List<PredictionHistoryResponse> getPredictionHistory() {
        return predictionRepository.findAllByOrderByCreatedAtDesc().stream()
                .map(p -> PredictionHistoryResponse.builder()
                        .id(p.getId())
                        .prediction(p.getPredictedClass())
                        .predictionDisplayName(p.getPredictionDisplayName())
                        .category(p.getCategory())
                        .riskLevel(p.getRiskLevel())
                        .confidence(p.getConfidence())
                        .modelName(p.getModelName())
                        .imagePath(p.getImagePath())
                        .createdAt(p.getCreatedAt())
                        .build())
                .collect(Collectors.toList());
    }

    public PredictionResponse getPredictionById(Long id) {
        Prediction p = predictionRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Prediction with ID " + id + " not found."));

        List<Map<String, Object>> topList = new ArrayList<>();
        try {
            if (p.getTopPredictionsJson() != null && !p.getTopPredictionsJson().isEmpty()) {
                topList = objectMapper.readValue(p.getTopPredictionsJson(), List.class);
            }
        } catch (Exception ignored) {}

        return PredictionResponse.builder()
                .predictionId(p.getId())
                .prediction(p.getPredictedClass())
                .predictionDisplayName(p.getPredictionDisplayName())
                .category(p.getCategory())
                .riskLevel(p.getRiskLevel())
                .confidence(p.getConfidence())
                .modelName(p.getModelName())
                .topPredictions(topList)
                .gradcamBase64(p.getGradcamBase64())
                .explanationAvailable(p.getGradcamBase64() != null)
                .timestamp(p.getCreatedAt())
                .build();
    }

    public void deletePrediction(Long id) {
        if (!predictionRepository.existsById(id)) {
            throw new ResourceNotFoundException("Prediction with ID " + id + " not found.");
        }
        predictionRepository.deleteById(id);
    }

    private void validateFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("Uploaded image file cannot be empty.");
        }
        String contentType = file.getContentType();
        if (contentType == null || (!contentType.startsWith("image/") && !contentType.equals("application/octet-stream"))) {
            throw new IllegalArgumentException("Invalid file format. File must be a valid image (JPEG, PNG, WEBP).");
        }
    }

    private String saveFileLocally(MultipartFile file) {
        try {
            Path dirPath = Paths.get(uploadDir);
            if (!Files.exists(dirPath)) {
                Files.createDirectories(dirPath);
            }
            String filename = System.currentTimeMillis() + "_" + (file.getOriginalFilename() != null ? file.getOriginalFilename() : "lesion.jpg");
            Path filePath = dirPath.resolve(filename);
            Files.copy(file.getInputStream(), filePath);
            return filePath.toString();
        } catch (IOException e) {
            return "uploads/lesion.jpg";
        }
    }
}
