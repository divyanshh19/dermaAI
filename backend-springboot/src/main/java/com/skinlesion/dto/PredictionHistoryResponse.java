package com.skinlesion.dto;

import lombok.*;
import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PredictionHistoryResponse {
    private Long id;
    private String prediction;
    private String predictionDisplayName;
    private String category;
    private String riskLevel;
    private Double confidence;
    private String modelName;
    private String imagePath;
    private LocalDateTime createdAt;
}
