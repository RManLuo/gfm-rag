from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "gfm-rag"


def read_script(name: str) -> str:
    return (SCRIPT_DIR / name).read_text()


def test_stage1_data_index_uses_current_workflow_entrypoint() -> None:
    content = read_script("stage1_data_index.sh")

    assert "python -m gfmrag.workflow.index_dataset" in content
    assert "gfmrag.workflow.stage1_index_dataset" not in content


def test_stage2_pretrain_uses_current_training_keys() -> None:
    content = read_script("stage2_pretrain.sh")

    assert "gfmrag.workflow.kgc_training" in content
    assert "trainer.fast_test=" in content
    assert "trainer.args.num_epoch=" in content
    assert "trainer.args.max_steps_per_epoch=" in content
    assert "trainer.args.train_batch_size=" in content
    assert "train.fast_test=" not in content
    assert "train.num_epoch=" not in content
    assert "train.batch_per_epoch=" not in content
    assert "train.batch_size=" not in content


def test_stage2_finetune_uses_current_training_entrypoint_and_keys() -> None:
    content = read_script("stage2_finetune.sh")

    assert "gfmrag.workflow.sft_training" in content
    assert "trainer.args.num_epoch=" in content
    assert "load_model_from_pretrained=" in content
    assert "trainer.args.do_train=false" in content
    assert "trainer.args.do_eval=true" in content
    assert "trainer.args.do_predict=true" in content
    assert "gfmrag.workflow.stage2_qa_finetune" not in content
    assert "train.num_epoch=" not in content
    assert "train.checkpoint=" not in content


def test_stage3_qa_inference_uses_current_inputs() -> None:
    content = read_script("stage3_qa_inference.sh")

    assert "python -m gfmrag.workflow.qa" in content
    assert "test.retrieved_result_path=" in content
    assert "test.node_path=" in content
    assert "gfmrag.workflow.stage3_qa_inference" not in content
    assert "dataset.root=" not in content
    assert "dataset.data_name=" not in content


def test_stage3_qa_ircot_inference_uses_nested_dataset_cfg_keys() -> None:
    content = read_script("stage3_qa_ircot_inference.sh")

    assert "python -m gfmrag.workflow.qa_ircot_inference" in content
    assert "dataset.cfgs.root=" in content
    assert "dataset.cfgs.data_name=" in content
    assert "graph_retriever.model_path=" in content
    assert "dataset.root=" not in content
    assert "dataset.data_name=" not in content
