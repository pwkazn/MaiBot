# Runtime image
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Working directory
WORKDIR /MaiMBot

ENV MAIBOT_LEGACY_0X_UPGRADE_CONFIRMED=1
ENV PATH="/MaiMBot/.venv/bin:${PATH}"

# Copy dependency metadata
COPY pyproject.toml uv.lock ./

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies
RUN uv sync --locked --no-dev --no-install-project

# Install system libraries required by Playwright Chromium. The browser binary
# itself is downloaded lazily into the configured data directory at runtime.
RUN python -m playwright install-deps chromium \
    && rm -rf /var/lib/apt/lists/*

# Copy project source
COPY . .

# 固定适配器提交，确保两个架构构建包含相同的结构化引用查询能力。
ARG NAPCAT_ADAPTER_REPOSITORY=https://github.com/pwkazn/MaiBot-Napcat-Adapter.git
ARG NAPCAT_ADAPTER_REF=24bae31202c87f3edf1913e8189b04250152efc5
RUN git clone --depth 1 --branch main "$NAPCAT_ADAPTER_REPOSITORY" plugin-templates/MaiBot-Napcat-Adapter \
    && git -C plugin-templates/MaiBot-Napcat-Adapter fetch --depth 1 origin "$NAPCAT_ADAPTER_REF" \
    && git -C plugin-templates/MaiBot-Napcat-Adapter checkout -B main FETCH_HEAD
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000 8001

ENTRYPOINT [ "./docker-entrypoint.sh" ]
