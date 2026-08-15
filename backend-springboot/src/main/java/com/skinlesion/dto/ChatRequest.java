package com.skinlesion.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.*;
import java.util.Map;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ChatRequest {

    @NotBlank(message = "Message content cannot be empty")
    private String message;

    private String conversationId;

    private Map<String, Object> predictionContext;
}
