.PHONY: build-core build-da3 shell-core shell-da3 test-core test-torch save-images load-images clean-docker-soft check

build-core:
	docker compose -f docker/docker-compose.yml build core

build-da3:
	docker compose -f docker/docker-compose.yml build da3

shell-core:
	docker compose -f docker/docker-compose.yml run --rm core

shell-da3:
	docker compose -f docker/docker-compose.yml run --rm --gpus all da3

test-core:
	docker compose -f docker/docker-compose.yml run --rm core python3 scripts/smoke_test_core.py

test-torch:
	docker compose -f docker/docker-compose.yml run --rm --gpus all da3 python3 scripts/smoke_test_torch.py

save-images:
	mkdir -p exports
	docker save construction-core-dev:latest -o exports/construction-core-dev.tar
	docker save construction-da3-dev:latest -o exports/construction-da3-dev.tar

load-images:
	docker load -i exports/construction-core-dev.tar
	docker load -i exports/construction-da3-dev.tar

clean-docker-soft:
	docker system prune -f

check:
	python3 scripts/check_env.py