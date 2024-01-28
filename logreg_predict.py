import sys
from Logistic_Regression import Logistic_Regression


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("ex) python logreg_predict.py \
              [TEST_DATA].csv [PARAMETER].pickle [OUTPUT].csv")
        exit(0)
    model = Logistic_Regression()
    model.get_parameter(sys.argv[2])
    model.predict_answer(sys.argv[1], sys.argv[3])
