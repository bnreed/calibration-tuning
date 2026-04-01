from pathlib import Path
from transformers import TrainerCallback
from llm.datasets import get_dataset, LMText, LabeledStringDataCollator
from llm.distributed import AcceleratorState
from llm.logging import entrypoint
from llm.models import get_model
from llm.models.peft import get_lora_model, get_temperature_head
from llm.trainer import WandbConfigUpdateCallback, CalibrationTuner


class ForceCheckpointCallback(TrainerCallback):
    """Force checkpoint saving at specified intervals."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.save_steps == 0:
            print(f"DEBUG ForceCheckpointCallback: Requesting save at step {state.global_step}")
            control.should_save = True
        return control


@entrypoint
def main(
    seed=137,
    log_dir=None,
    dataset=None,
    data_dir=None,
    prompt_style=None,
    max_token_length=None,
    num_workers=4,
    use_dataset_cache=True,
    model_name=None,
    use_local_model=False,
    model_dir=None,
    int8=True,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    peft_dir=None,
    ref_peft_dir=None,
    scale_temp=False,
    batch_size=1,
    gradient_accumulation_steps=1,
    lr=1e-4,
    warmup_ratio=0.0,
    kl_decay=0.0,
    max_steps=1,
    save_steps=None,
    resume_from_checkpoint=None,
    use_gradient_checkpointing=True,
    **_,
):
    if use_local_model and not model_dir:
        raise ValueError("When --use-local-model=True, you must set --model-dir.")

    accelerator = AcceleratorState()

    actual_save_steps = save_steps if save_steps is not None else max_steps // 10
    
    # When resuming, use the checkpoint's parent as output_dir
    if resume_from_checkpoint is not None and log_dir is None:
        log_dir = str(Path(resume_from_checkpoint).parent)
        print(f"DEBUG: Auto-setting log_dir to checkpoint parent: {log_dir}")
    
    print(f"DEBUG: save_steps parameter = {save_steps}")
    print(f"DEBUG: actual_save_steps = {actual_save_steps}")
    print(f"DEBUG: log_dir = {log_dir}")
    print(f"DEBUG: resume_from_checkpoint = {resume_from_checkpoint}")
    
    trainer_args = CalibrationTuner.Args(
        seed=seed,
        output_dir=log_dir,
        max_steps=max_steps,
        eval_steps=max_steps // 10,
        save_steps=actual_save_steps,
        logging_steps=max(1, max_steps // 200),
        dataloader_num_workers=num_workers,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        scale_temp=scale_temp,
        kl_decay=kl_decay,
    )
    
    print(f"DEBUG: trainer_args.save_steps = {trainer_args.save_steps}")
    print(f"DEBUG: trainer_args.save_strategy = {trainer_args.save_strategy}")

    with accelerator.main_process_first():
        train_data, val_data, test_data = get_dataset(
            dataset,
            root=data_dir,
            seed=seed,
            prompt_style=prompt_style,
            num_workers=num_workers,
            use_cache=use_dataset_cache,
        )
    if scale_temp:
        train_data, val_data = val_data, test_data or val_data

    # Fix: don't pass device_map for int8 quantization - let bitsandbytes handle it
    #if int8:
    #    tokenizer, model = get_model(
    ##       use_int8=int8,
    #   )
    #else:
    model_kwargs = {"model_dir": model_dir} if use_local_model else {}
    tokenizer, model = get_model(
        model_name,
        #device_map={"": accelerator.local_process_index},
        use_int8=int8,
        use_gradient_checkpointing=use_gradient_checkpointing,
        **model_kwargs,
    )

    if max_token_length is not None:
        tokenizer_args = LabeledStringDataCollator.get_tokenizer_args(tokenizer)

        def token_length_filter(instance):
            f_instance = {k: v for k, v in instance.items() if "embedding" not in k}
            inputs = tokenizer(
                [str(LMText.from_(f_instance))],
                **tokenizer_args,
            )
            return inputs.get("input_ids").size(-1) <= max_token_length

        train_data = train_data.filter(token_length_filter, num_proc=num_workers)
        val_data = val_data.filter(token_length_filter, num_proc=num_workers)

    model = get_lora_model(
        model,
        peft_id_or_dir=ref_peft_dir or peft_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        is_trainable=False,
        adapter_name="_ref",
    )

    model = get_lora_model(
        model,
        peft_id_or_dir=peft_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        is_trainable=not scale_temp,
        adapter_name="default",
    )

    if scale_temp:
        model.requires_grad_(False)

        temperature_model = get_temperature_head(
            checkpoint_dir=peft_dir,
            is_trainable=True,
            weights_name=CalibrationTuner.TEMPERATURE_WEIGHTS_NAME,
        ).to(accelerator.local_process_index)
    else:
        temperature_model = None
# BNR The HOTFIX comment mentioned it was "to allow registry with Trainer optimizer," but the trainer already has access to query_temperature_model through self.query_temperature_model, which is properly prepared by the accelerator in the _wrap_model method.


    trainer = CalibrationTuner(
        model=model,
        query_temperature_model=temperature_model,
        args=trainer_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        callbacks=[
            ForceCheckpointCallback(),
            WandbConfigUpdateCallback(
                dataset=dataset,
                prompt_style=prompt_style,
                max_token_length=max_token_length,
                model_name=model_name,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                peft_dir=peft_dir,
            ),
        ],
    )
    
    print(f"DEBUG: About to call trainer.train() with resume_from_checkpoint={resume_from_checkpoint}")
    if resume_from_checkpoint:
        import os
        print(f"DEBUG: Checkpoint exists? {os.path.exists(resume_from_checkpoint)}")
        if os.path.exists(resume_from_checkpoint):
            print(f"DEBUG: Checkpoint contents: {os.listdir(resume_from_checkpoint)}")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
