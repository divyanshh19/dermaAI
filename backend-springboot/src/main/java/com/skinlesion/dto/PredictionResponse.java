package com.skinlesion.dto;

import lombok.*;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PredictionResponse {
    private Long predictionId;
    private String prediction;
    private String predictionDisplayName;
    private String category;
    private String riskLevel;
    private Double confidence;
    private Boolean uncertain;
    private String uncertaintyMessage;
    private String modelName;
    private List<Map<String, Object>> topPredictions;
    private Map<String, Double> probabilities;
    private Map<String, Object> diseaseInfo;
    private Boolean explanationAvailable;
    private String gradcamBase64;
    private LocalDateTime timestamp;
}
