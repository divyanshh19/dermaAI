package com.skinlesion.service;

import com.skinlesion.client.NemotronClient;
import com.skinlesion.dto.*;
import com.skinlesion.entity.ChatMessage;
import com.skinlesion.entity.Conversation;
import com.skinlesion.exception.ResourceNotFoundException;
import com.skinlesion.repository.ChatMessageRepository;
import com.skinlesion.repository.ConversationRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ChatService {

    private final NemotronClient nemotronClient;
    private final ConversationRepository conversationRepository;
    private final ChatMessageRepository chatMessageRepository;

    public ChatResponse processChat(ChatRequest request) {
        String convId = request.getConversationId();
        if (convId == null || convId.trim().isEmpty()) {
            convId = UUID.randomUUID().toString();
            request.setConversationId(convId);
        }

        // 1. Get or create Conversation entity in MySQL
        String finalConvId = convId;
        Conversation conversation = conversationRepository.findById(finalConvId)
                .orElseGet(() -> conversationRepository.save(
                        Conversation.builder()
                                .id(finalConvId)
                                .userId("anonymous")
                                .build()
                ));

        // 2. Save user message to MySQL
        ChatMessage userMsg = ChatMessage.builder()
                .conversation(conversation)
                .role("user")
                .content(request.getMessage())
                .build();
        chatMessageRepository.save(userMsg);

        // 3. Call FastAPI Nemotron Chatbot microservice
        ChatResponse chatbotResponse = nemotronClient.chat(request);

        // 4. Save assistant response to MySQL
        ChatMessage assistantMsg = ChatMessage.builder()
                .conversation(conversation)
                .role("assistant")
                .content(chatbotResponse.getMessage())
                .build();
        chatMessageRepository.save(assistantMsg);

        // 5. Update conversation timestamp
        conversation.setUpdatedAt(LocalDateTime.now());
        conversationRepository.save(conversation);

        return chatbotResponse;
    }

    public ConversationResponse createConversation() {
        String newId = UUID.randomUUID().toString();
        Conversation conv = conversationRepository.save(
                Conversation.builder()
                        .id(newId)
                        .userId("anonymous")
                        .build()
        );
        return mapToConversationResponse(conv);
    }

    public List<ConversationResponse> getAllConversations() {
        return conversationRepository.findAllByOrderByUpdatedAtDesc().stream()
                .map(this::mapToConversationResponse)
                .collect(Collectors.toList());
    }

    public ConversationResponse getConversationById(String conversationId) {
        Conversation conv = conversationRepository.findById(conversationId)
                .orElseThrow(() -> new ResourceNotFoundException("Conversation " + conversationId + " not found."));
        return mapToConversationResponse(conv);
    }

    public void deleteConversation(String conversationId) {
        if (!conversationRepository.existsById(conversationId)) {
            throw new ResourceNotFoundException("Conversation " + conversationId + " not found.");
        }
        conversationRepository.deleteById(conversationId);
    }

    private ConversationResponse mapToConversationResponse(Conversation conv) {
        List<ChatMessageResponse> msgs = chatMessageRepository
                .findByConversationIdOrderByCreatedAtAsc(conv.getId())
                .stream()
                .map(m -> ChatMessageResponse.builder()
                        .id(m.getId())
                        .role(m.getRole())
                        .content(m.getContent())
                        .createdAt(m.getCreatedAt())
                        .build())
                .collect(Collectors.toList());

        return ConversationResponse.builder()
                .id(conv.getId())
                .userId(conv.getUserId())
                .createdAt(conv.getCreatedAt())
                .updatedAt(conv.getUpdatedAt())
                .messages(msgs)
                .build();
    }
}
