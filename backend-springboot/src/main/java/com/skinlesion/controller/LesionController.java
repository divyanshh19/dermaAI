package com.skinlesion.controller;

import com.skinlesion.dto.PredictionHistoryResponse;
import com.skinlesion.dto.PredictionResponse;
import com.skinlesion.service.PredictionService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@RestController
@RequestMapping("/api/lesions")
@RequiredArgsConstructor
@Tag(name = "Lesion Prediction & History", description = "Skin lesion classification, Grad-CAM heatmap generation, and MySQL prediction history endpoints")
public class LesionController {

    private final PredictionService predictionService;

    @PostMapping(value = "/predict", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "Classify skin lesion image & generate Grad-CAM heatmap", description = "Accepts a skin lesion image (JPEG, PNG, WEBP), sends it to PyTorch ML service, stores prediction in MySQL, and returns calibrated predictions.")
    public ResponseEntity<PredictionResponse> predict(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "explainable", defaultValue = "true") boolean explainable) {
        PredictionResponse response = predictionService.predictLesion(file, explainable);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/history")
    @Operation(summary = "Retrieve prediction history", description = "Returns all historical skin lesion predictions saved in MySQL.")
    public ResponseEntity<List<PredictionHistoryResponse>> getHistory() {
        return ResponseEntity.ok(predictionService.getPredictionHistory());
    }

    @GetMapping("/{id}")
    @Operation(summary = "Get prediction details by ID", description = "Retrieves a specific prediction record from MySQL.")
    public ResponseEntity<PredictionResponse> getById(@PathVariable("id") Long id) {
        return ResponseEntity.ok(predictionService.getPredictionById(id));
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "Delete prediction record by ID", description = "Deletes a specific prediction from MySQL.")
    public ResponseEntity<Void> deleteById(@PathVariable("id") Long id) {
        predictionService.deletePrediction(id);
        return ResponseEntity.noContent().build();
    }
}
