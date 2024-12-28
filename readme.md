
### **Description:**
python face recognition code based on sklearn lib, the code checks for identity of unknown person in its dataset, its using a train-tested model to do so, by default the train-test balance is 70% for train and 30% for testing

### Dev Environment:
- Python3.x (tested on 3.11)
- tested on ubuntu

### Usage:
- make VENV using this command : 
	 python3 -m venv env
- activate your VENV : 
	 win: env\scripts\activate
	 inux: source env/bin/activate
- install dependencies from requirements.txt :
	 pip install -r requirements.txt
- run combined.py via python :
	 **python3 combined.py -f test1.jpg**
	 the -f flag used to locate the testing image file