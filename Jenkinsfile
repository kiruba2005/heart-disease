pipeline {
    agent any

    stages {

        stage('Clone Repo') {
            steps {
                git 'https://github.com/kiruba2005/heart-disease.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install streamlit pandas scikit-learn numpy'
            }
        }

        stage('Run App') {
            steps {
                sh 'streamlit run app.py'
            }
        }
    }
}