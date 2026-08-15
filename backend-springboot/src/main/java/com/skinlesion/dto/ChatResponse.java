package com.skinlesion.dto;

import lombok.*;
import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ChatResponse {
    private String conversationId;
    private String message;
    private String model;
    private LocalDateTime timestamp;
}
