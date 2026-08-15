package com.skinlesion.client;

import com.skinlesion.dto.ChatRequest;
import com.skinlesion.dto.ChatResponse;
import com.skinlesion.exception.MlServiceException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

@Component
public class NemotronClient {

    private final RestTemplate restTemplate;

    @Value("${services.chatbot-service.url:http://localhost:8001}")
    private String chatbotServiceUrl;

    public NemotronClient() {
        this.restTemplate = new RestTemplate();
    }

    public ChatResponse chat(ChatRequest request) {
        String url = chatbotServiceUrl + "/chat";

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<ChatRequest> requestEntity = new HttpEntity<>(request, headers);

            ResponseEntity<ChatResponse> response = restTemplate.postForEntity(url, requestEntity, ChatResponse.class);

            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                return response.getBody();
            } else {
                throw new MlServiceException("Chatbot microservice returned status: " + response.getStatusCode());
            }
        } catch (Exception e) {
            throw new MlServiceException("Failed to communicate with Nemotron Chatbot Service at " + chatbotServiceUrl + ": " + e.getMessage(), e);
        }
    }
}
