NS = sgordon
TAG = latest

BASENAME = polycracker-miniconda

.PHONY : all
all : setup distribute clean push testrun test build

setup :
	python setup.py sdist bdist_wheel

distribute :
	twine upload dist/*

clean :
	docker system prune -f && rm -R dist && rm -R build

build_image : Dockerfile
	docker build -t $(NS)/$(BASENAME):$(TAG) .

push :
	docker push $(NS)/$(BASENAME):$(TAG)
