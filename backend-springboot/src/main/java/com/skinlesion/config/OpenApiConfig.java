package com.skinlesion.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.Contact;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class OpenApiConfig {

    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("AI Skin Lesion Detection System — Spring Boot API Gateway")
                        .version("3.0.0")
                        .description("Production Spring Boot 3.x REST API Gateway orchestrating PyTorch ML Inference & NVIDIA Nemotron AI Medical Assistant.")
                        .contact(new Contact().name("DermaAI Team")));
    }
}
