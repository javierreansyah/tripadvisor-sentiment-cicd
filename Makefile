install:
	pip install --upgrade pip &&\
		if [ -f Actions/Scripts/requirements.txt ]; then pip install -r Actions/Scripts/requirements.txt; fi

format:
	black Actions/Scripts/*.py

train:
	mkdir -p Actions/Model Actions/Results
	python Actions/Scripts/train_pipeline.py

eval:
	echo "## Model Metrics" > Actions/Results/report.md
	if [ -f ./Actions/Results/metrics.txt ]; then cat ./Actions/Results/metrics.txt >> Actions/Results/report.md; fi
	if [ -f ./Actions/Results/confusion_matrix.png ]; then \
		echo '\n## Evaluation Plot' >> Actions/Results/report.md; \
		echo '![Confusion Matrix](confusion_matrix.png)' >> Actions/Results/report.md; \
	fi
	cml comment create Actions/Results/report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add .
	git commit -am "Update with new results"
	git push --force origin HEAD:update

hf-login:
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload javierreansyah/Hotel-Review ./App . --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload javierreansyah/Hotel-Review ./Actions/Model/sentiment_pipeline.skops Actions/Model/sentiment_pipeline.skops --repo-type=space --commit-message="Sync Pipeline Model File"
	huggingface-cli upload javierreansyah/Hotel-Review ./Actions/Results/metrics.txt Actions/Results/metrics.txt --repo-type=space --commit-message="Sync Metrics File"

deploy: hf-login push-hub