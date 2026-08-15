package com.skinlesion.client;

import com.skinlesion.exception.MlServiceException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

@Component
public class MLServiceClient {

    private final RestTemplate restTemplate;

    @Value("${services.ml-service.url:http://localhost:8000}")
    private String mlServiceUrl;

    public MLServiceClient() {
        this.restTemplate = new RestTemplate();
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> predict(MultipartFile file, boolean explainable) {
        String url = mlServiceUrl + "/api/v1/predict?explainable=" + explainable;

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            ByteArrayResource fileResource = new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename() != null ? file.getOriginalFilename() : "lesion.jpg";
                }
            };

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", fileResource);

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            ResponseEntity<Map> response = restTemplate.postForEntity(url, requestEntity, Map.class);

            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                return (Map<String, Object>) response.getBody();
            } else {
                throw new MlServiceException("Python ML service returned status code: " + response.getStatusCode());
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to read uploaded image file.", e);
        } catch (Exception e) {
            throw new MlServiceException("Failed to communicate with Python ML Service at " + mlServiceUrl + ": " + e.getMessage(), e);
        }
    }
}
