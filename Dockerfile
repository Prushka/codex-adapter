# syntax=docker/dockerfile:1

ARG GO_VERSION=1.26

FROM --platform=$BUILDPLATFORM golang:${GO_VERSION}-alpine AS build

ARG TARGETOS
ARG TARGETARCH
ARG GIT_COMMIT=unknown
ARG GIT_VERSION=unknown

RUN echo "Building for OS: ${TARGETOS}, Architecture: ${TARGETARCH}, Commit: ${GIT_COMMIT}, Version: ${GIT_VERSION}"

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build \
    -trimpath \
    -ldflags="-s -w" \
    -o /out/load-balancer \
    ./cmd/load-balancer

FROM alpine:3.22 AS load-balancer

ARG GIT_COMMIT=unknown
ARG GIT_VERSION=unknown

LABEL org.opencontainers.image.title="meinya/llm-lb" \
    org.opencontainers.image.description="Chat Completions and Responses pass-through load balancer for codex-adapter" \
    org.opencontainers.image.ref.name="meinya/llm-lb" \
    org.opencontainers.image.revision=$GIT_COMMIT \
    org.opencontainers.image.version=$GIT_VERSION

RUN apk add --no-cache ca-certificates \
    && addgroup -S codex \
    && adduser -S -G codex -u 10001 codex

COPY --from=build /out/load-balancer /usr/local/bin/load-balancer

USER codex:codex
EXPOSE 18081

ENTRYPOINT ["/usr/local/bin/load-balancer"]
CMD ["-listen", "0.0.0.0:18081"]
