package com.skinlesion.service;

import com.skinlesion.client.MLServiceClient;
import com.skinlesion.dto.PredictionResponse;
import com.skinlesion.entity.Prediction;
import com.skinlesion.repository.PredictionRepository;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mock.web.MockMultipartFile;

import java.util.HashMap;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;

@ExtendWith(MockitoExtension.class)
public class PredictionServiceTest {

    @Mock
    private MLServiceClient mlServiceClient;

    @Mock
    private PredictionRepository predictionRepository;

    @InjectMocks
    private PredictionService predictionService;

    @Test
    public void testPredictLesionSuccess() {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "lesion.jpg",
                "image/jpeg",
                "dummy binary".getBytes()
        );

        Map<String, Object> mlResult = new HashMap<>();
        mlResult.put("prediction", "nv");
        mlResult.put("prediction_display_name", "Melanocytic Nevi");
        mlResult.put("category", "Benign");
        mlResult.put("risk_level", "Low Risk");
        mlResult.put("confidence", 0.95);
        mlResult.put("uncertain", false);
        mlResult.put("model_name", "EfficientNet-B0");

        Mockito.when(mlServiceClient.predict(any(), Mockito.eq(true))).thenReturn(mlResult);
        Mockito.when(predictionRepository.save(any())).thenAnswer(i -> {
            Prediction p = i.getArgument(0);
            p.setId(10L);
            return p;
        });

        PredictionResponse resp = predictionService.predictLesion(file, true);

        Assertions.assertNotNull(resp);
        Assertions.assertEquals("nv", resp.getPrediction());
        Assertions.assertEquals(0.95, resp.getConfidence());
        Assertions.assertEquals(10L, resp.getPredictionId());
    }

    @Test
    public void testPredictEmptyFileThrowsException() {
        MockMultipartFile emptyFile = new MockMultipartFile("file", "", "image/jpeg", new byte[0]);

        Assertions.assertThrows(IllegalArgumentException.class, () -> {
            predictionService.predictLesion(emptyFile, true);
        });
    }
}
