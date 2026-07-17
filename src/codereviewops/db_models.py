"""Append-oriented PostgreSQL persistence schema."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    metadata = metadata


class Timestamped:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class BenchmarkDefinition(Timestamped, Base):
    __tablename__ = "benchmark_definitions"
    __table_args__ = (
        UniqueConstraint("task_id", "content_hash"),
        CheckConstraint(
            "content_hash ~ '^sha256:[0-9a-f]{64}$'",
            name="content_hash_sha256",
        ),
        CheckConstraint("difficulty IN ('low','medium','high')", name="difficulty_values"),
        CheckConstraint(
            "primary_category IN "
            "('bug','requirement_mismatch','missing_test','performance','security',"
            "'maintainability','negative')",
            name="primary_category_values",
        ),
        CheckConstraint(
            "(negative AND primary_category = 'negative') OR "
            "(NOT negative AND primary_category <> 'negative')",
            name="negative_category_consistency",
        ),
        CheckConstraint("jsonb_typeof(tags) = 'array'", name="tags_array"),
        CheckConstraint("jsonb_typeof(task_snapshot) = 'object'", name="task_snapshot_object"),
        Index("ix_benchmark_definitions_task_active", "task_id", "is_active"),
        Index(
            "ix_benchmark_definitions_suite_active",
            "suite_id",
            "suite_version",
            "is_active",
        ),
        Index(
            "ix_benchmark_definitions_catalog_filters",
            "difficulty",
            "primary_category",
            "negative",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False)
    task_schema_version: Mapped[str] = mapped_column(String(16), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    suite_id: Mapped[str] = mapped_column(String(128), nullable=False)
    suite_version: Mapped[str] = mapped_column(String(64), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(16), nullable=False)
    primary_category: Mapped[str] = mapped_column(String(32), nullable=False)
    negative: Mapped[bool] = mapped_column(Boolean, nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    manifest_relative_path: Mapped[str] = mapped_column(String(512), nullable=False)
    task_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class ReviewRun(Timestamped, Base):
    __tablename__ = "review_runs"
    __table_args__ = (
        CheckConstraint("status IN ('running','completed','failed')", name="status_values"),
        CheckConstraint("provider IN ('replay','groq','mistral')", name="provider_values"),
        CheckConstraint(
            "tool_transport IN ('direct','mcp-stdio')",
            name="tool_transport_values",
        ),
        CheckConstraint(
            "(provider = 'replay' AND requested_model IS NULL) OR "
            "(provider IN ('groq','mistral') AND requested_model IS NOT NULL)",
            name="provider_model_consistency",
        ),
        CheckConstraint(
            "(status = 'running' AND completed_at IS NULL AND artifact_schema_version IS NULL "
            "AND artifact_snapshot IS NULL AND failure_code IS NULL "
            "AND failure_message IS NULL) OR "
            "(status = 'completed' AND completed_at IS NOT NULL "
            "AND artifact_schema_version IS NOT NULL AND artifact_snapshot IS NOT NULL "
            "AND failure_code IS NULL AND failure_message IS NULL) OR "
            "(status = 'failed' AND completed_at IS NOT NULL "
            "AND failure_code IS NOT NULL AND failure_message IS NOT NULL)",
            name="status_payload_consistency",
        ),
        CheckConstraint(
            "artifact_snapshot IS NULL OR jsonb_typeof(artifact_snapshot) = 'object'",
            name="artifact_snapshot_object",
        ),
        Index("ix_review_runs_status_created_id", "status", "created_at", "id"),
        Index("ix_review_runs_provider", "provider"),
        Index(
            "ix_review_runs_benchmark_created",
            "benchmark_definition_id",
            "created_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    benchmark_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("benchmark_definitions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    requested_model: Mapped[str | None] = mapped_column(String(128))
    tool_transport: Mapped[str] = mapped_column(String(16), nullable=False)
    artifact_schema_version: Mapped[str | None] = mapped_column(String(16))
    artifact_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_message: Mapped[str | None] = mapped_column(String(512))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class ReviewResult(Timestamped, Base):
    __tablename__ = "review_results"
    __table_args__ = (
        UniqueConstraint("run_id"),
        CheckConstraint(
            "assessment IN ('pass','needs_changes','fail','uncertain')",
            name="assessment_values",
        ),
        CheckConstraint(
            "precision >= 0 AND precision <= 1 "
            "AND recall >= 0 AND recall <= 1 "
            "AND hallucination_rate >= 0 AND hallucination_rate <= 1",
            name="rates_range",
        ),
        CheckConstraint(
            "matched_count >= 0 AND missed_count >= 0 AND hallucinated_count >= 0 "
            "AND prohibited_count >= 0",
            name="counts_nonnegative",
        ),
        CheckConstraint(
            "provider_latency_ms IS NULL OR "
            "(provider_latency_ms >= 0 AND provider_latency_ms < 'Infinity'::double precision)",
            name="provider_latency_finite_nonnegative",
        ),
        CheckConstraint(
            "(input_tokens IS NULL OR input_tokens >= 0) "
            "AND (output_tokens IS NULL OR output_tokens >= 0)",
            name="tokens_nonnegative",
        ),
        CheckConstraint("char_length(summary) <= 65536", name="summary_bounded"),
        CheckConstraint("jsonb_typeof(limitations) = 'array'", name="limitations_array"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("review_runs.id", ondelete="RESTRICT"), nullable=False
    )
    assessment: Mapped[str] = mapped_column(String(32), nullable=False)
    task_success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    precision: Mapped[float] = mapped_column(Double, nullable=False)
    recall: Mapped[float] = mapped_column(Double, nullable=False)
    hallucination_rate: Mapped[float] = mapped_column(Double, nullable=False)
    matched_count: Mapped[int] = mapped_column(Integer, nullable=False)
    missed_count: Mapped[int] = mapped_column(Integer, nullable=False)
    hallucinated_count: Mapped[int] = mapped_column(Integer, nullable=False)
    prohibited_count: Mapped[int] = mapped_column(Integer, nullable=False)
    provider_latency_ms: Mapped[float | None] = mapped_column(Double)
    input_tokens: Mapped[int | None] = mapped_column(BigInteger)
    output_tokens: Mapped[int | None] = mapped_column(BigInteger)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    limitations: Mapped[list[str]] = mapped_column(JSONB, nullable=False)


class Finding(Timestamped, Base):
    __tablename__ = "findings"
    __table_args__ = (
        UniqueConstraint("run_id", "finding_index"),
        CheckConstraint("finding_index >= 0", name="finding_index_nonnegative"),
        UniqueConstraint("run_id", "id"),
        CheckConstraint(
            "severity IN ('low','medium','high','critical')",
            name="severity_values",
        ),
        CheckConstraint(
            "category IN "
            "('bug','requirement_mismatch','missing_test','performance','security','maintainability')",
            name="category_values",
        ),
        CheckConstraint(
            "line_start > 0 AND line_end > 0 AND line_end >= line_start",
            name="line_range_valid",
        ),
        CheckConstraint(
            "char_length(file) > 0 AND position(chr(92) in file) = 0 "
            "AND file !~ '^/' AND file !~ '^[A-Za-z]:' "
            "AND file !~ '(^|/)\\.\\.(/|$)'",
            name="file_relative",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="confidence_range",
        ),
        CheckConstraint(
            "evaluation_disposition IN ('matched','hallucinated')",
            name="evaluation_disposition_values",
        ),
        CheckConstraint(
            "(evaluation_disposition = 'matched' AND matched_expected_index IS NOT NULL "
            "AND matched_expected_index >= 0) OR "
            "(evaluation_disposition = 'hallucinated' AND matched_expected_index IS NULL)",
            name="matched_expected_consistency",
        ),
        CheckConstraint("char_length(title) <= 256", name="title_bounded"),
        CheckConstraint("char_length(evidence) <= 65536", name="evidence_bounded"),
        CheckConstraint("char_length(reasoning) <= 65536", name="reasoning_bounded"),
        CheckConstraint("char_length(recommendation) <= 65536", name="recommendation_bounded"),
        CheckConstraint("jsonb_typeof(evidence_trace_ids) = 'array'", name="trace_ids_array"),
        Index("ix_findings_run_category_severity", "run_id", "category", "severity"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("review_runs.id", ondelete="RESTRICT"), nullable=False
    )
    finding_index: Mapped[int] = mapped_column(Integer, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    file: Mapped[str] = mapped_column(String(512), nullable=False)
    line_start: Mapped[int] = mapped_column(Integer, nullable=False)
    line_end: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Double, nullable=False)
    evaluation_disposition: Mapped[str] = mapped_column(String(16), nullable=False)
    matched_expected_index: Mapped[int | None] = mapped_column(Integer)
    evidence_trace_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)


class ToolTrace(Timestamped, Base):
    __tablename__ = "tool_traces"
    __table_args__ = (
        UniqueConstraint("run_id", "trace_id"),
        UniqueConstraint("run_id", "trace_order"),
        CheckConstraint("trace_order >= 0", name="trace_order_nonnegative"),
        CheckConstraint(
            "tool IN ('read_file','search_code','run_tests')",
            name="tool_values",
        ),
        CheckConstraint("status IN ('succeeded','failed')", name="status_values"),
        CheckConstraint(
            "latency_ms >= 0 AND latency_ms < 'Infinity'::double precision",
            name="latency_finite_nonnegative",
        ),
        CheckConstraint("jsonb_typeof(arguments) = 'object'", name="arguments_object"),
        CheckConstraint("jsonb_typeof(result) = 'object'", name="result_object"),
        CheckConstraint("jsonb_typeof(influence) = 'object'", name="influence_object"),
        CheckConstraint("jsonb_typeof(provenance) = 'array'", name="provenance_array"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("review_runs.id", ondelete="RESTRICT"), nullable=False
    )
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    trace_order: Mapped[int] = mapped_column(Integer, nullable=False)
    tool: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Double, nullable=False)
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    influence: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    provenance: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)


class HumanFeedback(Timestamped, Base):
    __tablename__ = "human_feedback"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('correct','incorrect','missed','note')",
            name="kind_values",
        ),
        CheckConstraint(
            "(kind IN ('correct','incorrect') AND finding_id IS NOT NULL) OR "
            "(kind IN ('missed','note') AND finding_id IS NULL)",
            name="kind_finding_consistency",
        ),
        ForeignKeyConstraint(
            ["run_id", "finding_id"],
            ["findings.run_id", "findings.id"],
            ondelete="RESTRICT",
        ),
        Index("ix_human_feedback_run_created_id", "run_id", "created_at", "id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("review_runs.id", ondelete="RESTRICT"), nullable=False
    )
    finding_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    comment: Mapped[str | None] = mapped_column(String(2000))
    reviewer_reference: Mapped[str | None] = mapped_column(String(256))


class IdempotencyRecord(Timestamped, Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        UniqueConstraint("endpoint_scope", "key"),
        CheckConstraint(
            "request_hash ~ '^sha256:[0-9a-f]{64}$'",
            name="request_hash_sha256",
        ),
        CheckConstraint(
            "(run_id IS NOT NULL AND feedback_id IS NULL) OR "
            "(run_id IS NULL AND feedback_id IS NOT NULL)",
            name="exactly_one_resource",
        ),
        Index("ix_idempotency_records_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    endpoint_scope: Mapped[str] = mapped_column(String(64), nullable=False)
    key: Mapped[str] = mapped_column(String(256), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(71), nullable=False)
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("review_runs.id", ondelete="RESTRICT")
    )
    feedback_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("human_feedback.id", ondelete="RESTRICT")
    )
