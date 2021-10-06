from regression_module import Regression_module

if __name__ == '__main__':
    model = Regression_module()
    model.train("train.csv")
    print(model.r2_test())
    answer=model.predict("test.csv")
    answer.to_csv("test1.csv", index=False)




