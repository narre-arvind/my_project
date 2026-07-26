from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == '__main__':
    # Run ingestion
    di = DataIngestion()
    train_path, test_path = di.initiate_data_ingestion()
    print('Train path:', train_path)
    print('Test path :', test_path)

    # Run transformation (target column: math_score)
    dt = DataTransformation()
    train_arr, test_arr, preprocessor_path = dt.initiate_data_transformation(
        train_path=train_path,
        test_path=test_path,
        target_column_name='math_score'
    )

    print('Preprocessor saved at:', preprocessor_path)
    print('Train array shape:', train_arr.shape)
    print('Test array shape :', test_arr.shape)
