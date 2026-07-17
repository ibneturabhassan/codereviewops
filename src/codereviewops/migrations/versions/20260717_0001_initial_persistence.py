"""Create the initial append-oriented persistence schema."""
# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260717_0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB()
NOW = sa.text("now()")


def col(name: str, kind: Any, nullable: bool = False, default: Any = None) -> sa.Column[Any]:
    return sa.Column(name, kind, nullable=nullable, server_default=default)


def base() -> list[Any]:
    return [col("id", UUID), col("created_at", sa.DateTime(timezone=True), default=NOW)]


def ck(_table: str, name: str, expression: str) -> sa.CheckConstraint:
    return sa.CheckConstraint(expression, name=name)


def upgrade() -> None:
    op.create_table(
        "benchmark_definitions",
        *base(),
        col("task_id", sa.String(128)),
        col("task_schema_version", sa.String(16)),
        col("content_hash", sa.String(71)),
        col("suite_id", sa.String(128)),
        col("suite_version", sa.String(64)),
        col("difficulty", sa.String(16)),
        col("primary_category", sa.String(32)),
        col("negative", sa.Boolean()),
        col("tags", JSONB),
        col("manifest_relative_path", sa.String(512)),
        col("task_snapshot", JSONB),
        col("is_active", sa.Boolean()),
        ck(
            "benchmark_definitions", "content_hash_sha256", "content_hash ~ '^sha256:[0-9a-f]{64}$'"
        ),
        ck("benchmark_definitions", "difficulty_values", "difficulty IN ('low','medium','high')"),
        ck(
            "benchmark_definitions",
            "primary_category_values",
            "primary_category IN ('bug','requirement_mismatch','missing_test','performance','security','maintainability','negative')",
        ),
        ck(
            "benchmark_definitions",
            "negative_category_consistency",
            "(negative AND primary_category = 'negative') OR (NOT negative AND primary_category <> 'negative')",
        ),
        ck("benchmark_definitions", "tags_array", "jsonb_typeof(tags) = 'array'"),
        ck(
            "benchmark_definitions",
            "task_snapshot_object",
            "jsonb_typeof(task_snapshot) = 'object'",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_benchmark_definitions"),
        sa.UniqueConstraint(
            "task_id", "content_hash", name="uq_benchmark_definitions_task_id_content_hash"
        ),
    )
    op.create_index(
        "ix_benchmark_definitions_task_active", "benchmark_definitions", ["task_id", "is_active"]
    )
    op.create_index(
        "ix_benchmark_definitions_suite_active",
        "benchmark_definitions",
        ["suite_id", "suite_version", "is_active"],
    )
    op.create_index(
        "ix_benchmark_definitions_catalog_filters",
        "benchmark_definitions",
        ["difficulty", "primary_category", "negative"],
    )

    op.create_table(
        "review_runs",
        *base(),
        col("benchmark_definition_id", UUID),
        col("status", sa.String(16)),
        col("provider", sa.String(16)),
        col("requested_model", sa.String(128), True),
        col("tool_transport", sa.String(16)),
        col("artifact_schema_version", sa.String(16), True),
        col("artifact_snapshot", JSONB, True),
        col("failure_code", sa.String(64), True),
        col("failure_message", sa.String(512), True),
        col("started_at", sa.DateTime(timezone=True), default=NOW),
        col("completed_at", sa.DateTime(timezone=True), True),
        col("updated_at", sa.DateTime(timezone=True), default=NOW),
        ck("review_runs", "status_values", "status IN ('running','completed','failed')"),
        ck("review_runs", "provider_values", "provider IN ('replay','groq','mistral')"),
        ck("review_runs", "tool_transport_values", "tool_transport IN ('direct','mcp-stdio')"),
        ck(
            "review_runs",
            "provider_model_consistency",
            "(provider = 'replay' AND requested_model IS NULL) OR (provider IN ('groq','mistral') AND requested_model IS NOT NULL)",
        ),
        ck(
            "review_runs",
            "status_payload_consistency",
            "(status = 'running' AND completed_at IS NULL AND artifact_schema_version IS NULL AND artifact_snapshot IS NULL AND failure_code IS NULL AND failure_message IS NULL) OR (status = 'completed' AND completed_at IS NOT NULL AND artifact_schema_version IS NOT NULL AND artifact_snapshot IS NOT NULL AND failure_code IS NULL AND failure_message IS NULL) OR (status = 'failed' AND completed_at IS NOT NULL AND failure_code IS NOT NULL AND failure_message IS NOT NULL)",
        ),
        ck(
            "review_runs",
            "artifact_snapshot_object",
            "artifact_snapshot IS NULL OR jsonb_typeof(artifact_snapshot) = 'object'",
        ),
        sa.ForeignKeyConstraint(
            ["benchmark_definition_id"],
            ["benchmark_definitions.id"],
            name="fk_review_runs_benchmark_definition_id_benchmark_definitions",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_review_runs"),
    )
    op.create_index(
        "ix_review_runs_status_created_id", "review_runs", ["status", "created_at", "id"]
    )
    op.create_index("ix_review_runs_provider", "review_runs", ["provider"])
    op.create_index(
        "ix_review_runs_benchmark_created", "review_runs", ["benchmark_definition_id", "created_at"]
    )

    op.create_table(
        "review_results",
        *base(),
        col("run_id", UUID),
        col("assessment", sa.String(32)),
        col("task_success", sa.Boolean()),
        col("precision", sa.Double()),
        col("recall", sa.Double()),
        col("hallucination_rate", sa.Double()),
        col("matched_count", sa.Integer()),
        col("missed_count", sa.Integer()),
        col("hallucinated_count", sa.Integer()),
        col("prohibited_count", sa.Integer()),
        col("provider_latency_ms", sa.Double(), True),
        col("input_tokens", sa.BigInteger(), True),
        col("output_tokens", sa.BigInteger(), True),
        col("summary", sa.Text()),
        col("limitations", JSONB),
        ck(
            "review_results",
            "assessment_values",
            "assessment IN ('pass','needs_changes','fail','uncertain')",
        ),
        ck(
            "review_results",
            "rates_range",
            "precision >= 0 AND precision <= 1 AND recall >= 0 AND recall <= 1 AND hallucination_rate >= 0 AND hallucination_rate <= 1",
        ),
        ck(
            "review_results",
            "counts_nonnegative",
            "matched_count >= 0 AND missed_count >= 0 AND hallucinated_count >= 0 AND prohibited_count >= 0",
        ),
        ck(
            "review_results",
            "provider_latency_finite_nonnegative",
            "provider_latency_ms IS NULL OR (provider_latency_ms >= 0 AND provider_latency_ms < 'Infinity'::double precision)",
        ),
        ck(
            "review_results",
            "tokens_nonnegative",
            "(input_tokens IS NULL OR input_tokens >= 0) AND (output_tokens IS NULL OR output_tokens >= 0)",
        ),
        ck("review_results", "summary_bounded", "char_length(summary) <= 65536"),
        ck("review_results", "limitations_array", "jsonb_typeof(limitations) = 'array'"),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name="fk_review_results_run_id_review_runs",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_review_results"),
        sa.UniqueConstraint("run_id", name="uq_review_results_run_id"),
    )

    op.create_table(
        "findings",
        *base(),
        col("run_id", UUID),
        col("finding_index", sa.Integer()),
        col("severity", sa.String(16)),
        col("category", sa.String(32)),
        col("file", sa.String(512)),
        col("line_start", sa.Integer()),
        col("line_end", sa.Integer()),
        col("title", sa.Text()),
        col("evidence", sa.Text()),
        col("reasoning", sa.Text()),
        col("recommendation", sa.Text()),
        col("confidence", sa.Double()),
        col("evaluation_disposition", sa.String(16)),
        col("matched_expected_index", sa.Integer(), True),
        col("evidence_trace_ids", JSONB),
        ck("findings", "finding_index_nonnegative", "finding_index >= 0"),
        ck("findings", "severity_values", "severity IN ('low','medium','high','critical')"),
        ck(
            "findings",
            "category_values",
            "category IN ('bug','requirement_mismatch','missing_test','performance','security','maintainability')",
        ),
        ck(
            "findings",
            "line_range_valid",
            "line_start > 0 AND line_end > 0 AND line_end >= line_start",
        ),
        ck(
            "findings",
            "file_relative",
            "char_length(file) > 0 AND position(chr(92) in file) = 0 AND file !~ '^/' AND file !~ '^[A-Za-z]:' AND file !~ '(^|/)\\.\\.(/|$)'",
        ),
        ck("findings", "confidence_range", "confidence >= 0 AND confidence <= 1"),
        ck(
            "findings",
            "evaluation_disposition_values",
            "evaluation_disposition IN ('matched','hallucinated')",
        ),
        ck(
            "findings",
            "matched_expected_consistency",
            "(evaluation_disposition = 'matched' AND matched_expected_index IS NOT NULL AND matched_expected_index >= 0) OR (evaluation_disposition = 'hallucinated' AND matched_expected_index IS NULL)",
        ),
        ck("findings", "title_bounded", "char_length(title) <= 256"),
        ck("findings", "evidence_bounded", "char_length(evidence) <= 65536"),
        ck("findings", "reasoning_bounded", "char_length(reasoning) <= 65536"),
        ck("findings", "recommendation_bounded", "char_length(recommendation) <= 65536"),
        ck("findings", "trace_ids_array", "jsonb_typeof(evidence_trace_ids) = 'array'"),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name="fk_findings_run_id_review_runs",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_findings"),
        sa.UniqueConstraint("run_id", "finding_index", name="uq_findings_run_id_finding_index"),
        sa.UniqueConstraint("run_id", "id", name="uq_findings_run_id_id"),
    )
    op.create_index(
        "ix_findings_run_category_severity", "findings", ["run_id", "category", "severity"]
    )

    op.create_table(
        "tool_traces",
        *base(),
        col("run_id", UUID),
        col("trace_id", sa.String(128)),
        col("trace_order", sa.Integer()),
        col("tool", sa.String(32)),
        col("status", sa.String(16)),
        col("latency_ms", sa.Double()),
        col("arguments", JSONB),
        col("result", JSONB),
        col("influence", JSONB),
        col("provenance", JSONB),
        ck("tool_traces", "trace_order_nonnegative", "trace_order >= 0"),
        ck("tool_traces", "tool_values", "tool IN ('read_file','search_code','run_tests')"),
        ck("tool_traces", "status_values", "status IN ('succeeded','failed')"),
        ck(
            "tool_traces",
            "latency_finite_nonnegative",
            "latency_ms >= 0 AND latency_ms < 'Infinity'::double precision",
        ),
        ck("tool_traces", "arguments_object", "jsonb_typeof(arguments) = 'object'"),
        ck("tool_traces", "result_object", "jsonb_typeof(result) = 'object'"),
        ck("tool_traces", "influence_object", "jsonb_typeof(influence) = 'object'"),
        ck("tool_traces", "provenance_array", "jsonb_typeof(provenance) = 'array'"),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name="fk_tool_traces_run_id_review_runs",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_tool_traces"),
        sa.UniqueConstraint("run_id", "trace_id", name="uq_tool_traces_run_id_trace_id"),
        sa.UniqueConstraint("run_id", "trace_order", name="uq_tool_traces_run_id_trace_order"),
    )

    op.create_table(
        "human_feedback",
        *base(),
        col("run_id", UUID),
        col("finding_id", UUID, True),
        col("kind", sa.String(16)),
        col("comment", sa.String(2000), True),
        col("reviewer_reference", sa.String(256), True),
        ck("human_feedback", "kind_values", "kind IN ('correct','incorrect','missed','note')"),
        ck(
            "human_feedback",
            "kind_finding_consistency",
            "(kind IN ('correct','incorrect') AND finding_id IS NOT NULL) OR (kind IN ('missed','note') AND finding_id IS NULL)",
        ),
        sa.ForeignKeyConstraint(
            ["run_id", "finding_id"],
            ["findings.run_id", "findings.id"],
            name="fk_human_feedback_run_id_finding_id_findings",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name="fk_human_feedback_run_id_review_runs",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_human_feedback"),
    )
    op.create_index(
        "ix_human_feedback_run_created_id", "human_feedback", ["run_id", "created_at", "id"]
    )

    op.create_table(
        "idempotency_records",
        *base(),
        col("endpoint_scope", sa.String(64)),
        col("key", sa.String(256)),
        col("request_hash", sa.String(71)),
        col("run_id", UUID, True),
        col("feedback_id", UUID, True),
        ck("idempotency_records", "request_hash_sha256", "request_hash ~ '^sha256:[0-9a-f]{64}$'"),
        ck(
            "idempotency_records",
            "exactly_one_resource",
            "(run_id IS NOT NULL AND feedback_id IS NULL) OR (run_id IS NULL AND feedback_id IS NOT NULL)",
        ),
        sa.ForeignKeyConstraint(
            ["feedback_id"],
            ["human_feedback.id"],
            name="fk_idempotency_records_feedback_id_human_feedback",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["review_runs.id"],
            name="fk_idempotency_records_run_id_review_runs",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_idempotency_records"),
        sa.UniqueConstraint(
            "endpoint_scope", "key", name="uq_idempotency_records_endpoint_scope_key"
        ),
    )
    op.create_index("ix_idempotency_records_created_at", "idempotency_records", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_idempotency_records_created_at", table_name="idempotency_records")
    op.drop_table("idempotency_records")
    op.drop_index("ix_human_feedback_run_created_id", table_name="human_feedback")
    op.drop_table("human_feedback")
    op.drop_table("tool_traces")
    op.drop_index("ix_findings_run_category_severity", table_name="findings")
    op.drop_table("findings")
    op.drop_table("review_results")
    op.drop_index("ix_review_runs_benchmark_created", table_name="review_runs")
    op.drop_index("ix_review_runs_provider", table_name="review_runs")
    op.drop_index("ix_review_runs_status_created_id", table_name="review_runs")
    op.drop_table("review_runs")
    op.drop_index("ix_benchmark_definitions_catalog_filters", table_name="benchmark_definitions")
    op.drop_index("ix_benchmark_definitions_suite_active", table_name="benchmark_definitions")
    op.drop_index("ix_benchmark_definitions_task_active", table_name="benchmark_definitions")
    op.drop_table("benchmark_definitions")
