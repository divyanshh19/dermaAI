package com.skinlesion.entity;

import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "predictions")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String imagePath;

    @Column(nullable = false)
    private String predictedClass;

    @Column(nullable = false)
    private String predictionDisplayName;

    @Column(nullable = false)
    private Double confidence;

    private String category;

    private String riskLevel;

    @Column(nullable = false)
    private String modelName;

    @Column(length = 2000)
    private String topPredictionsJson;

    @Column(columnDefinition = "LONGTEXT")
    private String gradcamBase64;

    @Column(nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        this.createdAt = LocalDateTime.now();
    }
}
