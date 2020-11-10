pipeline {
    agent {
        docker {
            // Important to use the `devel` version since the `runtime` version does not include e.g. compilers.
            image 'pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel'
            args '--user 0:0 --gpus all'
        }
    }
    environment {
        DISABLE_AUTH = 'true'
        DB_ENGINE    = 'sqlite'
    }
    stages {
        stage('test-cuda') {
            steps {
                sh """ python -c "import torch; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Using device:', device); torch.rand(10).to(device)" """
            }
        }
        stage('download-dependencies') {
            steps {
                sh 'pip install pytest pytest-cov coverage numpydoc flake8 codecov'
            }
        }
        stage('build') {
            steps {
                sh 'pip install --editable ./keops/'
                sh 'pip install --verbose --editable .[test]'
            }
        }
        stage('test') {
            steps {
                sh 'pytest --cov-report=term-missing --cov=falkon --cov-config setup.cfg'
                sh 'flake8 --count falkon'
            }
        }
    }
    post {
        cleanup {
            cleanWs()
        }
    }
}
