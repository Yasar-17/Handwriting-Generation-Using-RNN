.PHONY: help install test verify sanity train train-conditioned train-gan inference export docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make test             - Run all tests"
	@echo "  make verify           - Run model verification"
	@echo "  make sanity           - Run data pipeline sanity check"
	@echo "  make generate-data    - Generate synthetic training data"
	@echo "  make train            - Train unconditional model"
	@echo "  make train-conditioned - Train conditioned model"
	@echo "  make train-gan        - Train with GAN refinement"
	@echo "  make inference        - Run inference on trained model"
	@echo "  make export           - Export model to TorchScript/ONNX"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo "  make clean            - Remove generated files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

verify:
	python verify_model.py

sanity:
	python sanity_check.py

generate-data:
	python generate_synthetic_data.py --output_dir ./synthetic_data --num_samples 500

train: generate-data
	python train.py \
		--data_dir ./synthetic_data \
		--output_dir ./output \
		--epochs 50 --batch_size 64 \
		--hidden_dim 256 --num_layers 3 --num_mixtures 20 \
		--sample_every 5 --temperature 0.5 \
		--seed 42

train-conditioned: generate-data
	python train.py --conditioned \
		--data_dir ./synthetic_data \
		--output_dir ./output_conditioned \
		--epochs 50 --batch_size 64 \
		--hidden_dim 256 --num_layers 3 --num_mixtures 20 \
		--num_windows 10 --char_embed_dim 32 \
		--condition_text "the quick brown fox" \
		--sample_every 5 --temperature 0.5 \
		--warmup_epochs 5 --use_cosine_annealing \
		--seed 42

train-gan: generate-data
	python train.py --conditioned --use_gan \
		--data_dir ./synthetic_data \
		--output_dir ./output_conditioned_gan \
		--epochs 50 --batch_size 32 \
		--hidden_dim 256 --num_layers 3 --num_mixtures 20 \
		--num_windows 10 --char_embed_dim 32 \
		--condition_text "the quick brown fox" \
		--sample_every 5 --temperature 0.5 \
		--adv_weight 0.1 --disc_lr 1e-4 \
		--use_amp --grad_accum_steps 2 \
		--warmup_epochs 5 --early_stopping_patience 10 \
		--seed 42

inference:
	python inference.py \
		--ckpt ./output_conditioned/checkpoints/checkpoint_best.pt \
		--stats ./output_conditioned/stats.json \
		--text "hello world" "the quick brown fox" \
		--output_dir ./generated \
		--temperature 0.5 --num_samples 3

export:
	python export_model.py \
		--ckpt_path ./output_conditioned/checkpoints/checkpoint_best.pt \
		--output_dir ./exported_model \
		--stats_path ./output_conditioned/stats.json \
		--format both

docker-build:
	docker build -t handwriting-rnn-gan .

docker-run:
	docker run --rm -v $(PWD)/output:/app/output handwriting-rnn-gan python train.py --help

tensorboard:
	tensorboard --logdir ./output/tensorboard --port 6006

clean:
	rm -rf __pycache__ */__pycache__
	rm -rf output/ output_conditioned/ output_conditioned_gan/
	rm -rf sanity_output/ eval_output/ generated/ exported_model/
	rm -rf synthetic_data/
	find . -name "*.pt" -not -path "./.git/*" -delete
