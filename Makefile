

sync: .FORCE
	cp -r _action ../ml-demo/
	cp -r .github ../ml-demo/
	cp .gitignore ../ml-demo/
	cp train.py ../ml-demo/
	cp README.md ../ml-demo/

.FORCE:
