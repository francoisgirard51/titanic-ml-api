# Makefile

# Docker image name and port
IMAGE_NAME = titanic-api
PORT = 8000

# Show help
help:
	@echo "Available commands:"
	@echo "  make build            Build Docker image"
	@echo "  make run              Run API container (port $(PORT))"
	@echo "  make run-with-volume  Run container with mounted volume for models/"
	@echo "  make predict          Send test request to API"
	@echo "  make stop             Stop container using port $(PORT)"
	@echo "  make rebuild          Stop + build image"
	@echo "  make clean            Remove image and prune stopped containers"

# Build Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run API container
run:
	docker run -p $(PORT):8000 $(IMAGE_NAME)

# Run with local model volume
run-with-volume:
	docker run -v $(PWD)/models:/app/models -p $(PORT):8000 $(IMAGE_NAME)

# Test prediction
predict:
	curl -X POST http://127.0.0.1:$(PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"Pclass":3,"Sex":"male","Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":"S"}'

# Stop container using port $(PORT)
stop:
	@docker ps --filter "ancestor=$(IMAGE_NAME)" --format "{{.ID}}" | xargs -r docker stop
	@echo "✅ Container(s) on port $(PORT) stopped."

# Rebuild image after stop
rebuild: stop build

# Clean all containers & image
clean:
	@docker container prune -f
	@docker image rm -f $(IMAGE_NAME)
	@echo "🧹 Docker cleanup done."
