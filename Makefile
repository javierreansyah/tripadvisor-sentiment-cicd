install:
	pip install --upgrade pip &&\
		if [ -f requirements.txt ]; then pip install -r requirements.txt; fi &&\
		python -m nltk.downloader punkt stopwords wordnet

format:
	black Scripts/*.py

train:
	mkdir -p Model Results
	python Scripts/train_pipeline.py

eval:
	echo "## Model Metrics" > Results/report.md
	if [ -f ./Results/metrics.txt ]; then cat ./Results/metrics.txt >> Results/report.md; fi
	if [ -f ./Results/confusion_matrix.png ]; then \
		echo '\n## Evaluation Plot' >> Results/report.md; \
		echo '![Confusion Matrix](confusion_matrix.png)' >> Results/report.md; \
	fi
	cml comment create Results/report.md

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
	huggingface-cli upload javierreansyah/Hotel-Review ./Model/logreg_tfidf.skops Model/logreg_tfidf.skops --repo-type=space --commit-message="Sync Model File"
	huggingface-cli upload javierreansyah/Hotel-Review ./Model/tfidf_vectorizer.skops Model/tfidf_vectorizer.skops --repo-type=space --commit-message="Sync Vectorizer File"
	huggingface-cli upload javierreansyah/Hotel-Review ./Results/metrics.txt Results/metrics.txt --repo-type=space --commit-message="Sync Metrics File"

deploy: hf-login push-hub