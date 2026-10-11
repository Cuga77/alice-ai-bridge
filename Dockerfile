FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /alice-ai-bridge .

# distroless/static: CA certificates for HTTPS providers, no shell, runs as non-root.
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /alice-ai-bridge /alice-ai-bridge
EXPOSE 8080
ENTRYPOINT ["/alice-ai-bridge"]
