package com.skinlesion.controller;

import com.skinlesion.dto.PredictionResponse;
import com.skinlesion.service.PredictionService;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;
import java.util.Collections;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(LesionController.class)
public class LesionControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private PredictionService predictionService;

    @Test
    public void testPredictSuccess() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "test.jpg",
                "image/jpeg",
                "fake image content".getBytes()
        );

        PredictionResponse mockResponse = PredictionResponse.builder()
                .predictionId(1L)
                .prediction("melanoma")
                .predictionDisplayName("Melanoma")
                .category("Malignant Skin Cancer")
                .riskLevel("Critical Risk")
                .confidence(0.92)
                .modelName("EfficientNet-B0")
                .topPredictions(Collections.emptyList())
                .timestamp(LocalDateTime.now())
                .build();

        Mockito.when(predictionService.predictLesion(any(), eq(true))).thenReturn(mockResponse);

        mockMvc.perform(multipart("/api/lesions/predict").file(file).param("explainable", "true"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.prediction").value("melanoma"))
                .andExpect(jsonPath("$.confidence").value(0.92))
                .andExpect(jsonPath("$.predictionId").value(1));
    }

    @Test
    public void testGetHistorySuccess() throws Exception {
        Mockito.when(predictionService.getPredictionHistory()).thenReturn(Collections.emptyList());

        mockMvc.perform(get("/api/lesions/history"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray());
    }
}
