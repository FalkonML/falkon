def getGitCommit() {
    return sh(script: "git log -1 --pretty=%B", returnStdout: true)
}

def getCommitTag() {
    return sh(
        returnStdout: true,
        script: 'git fetch --tags && git tag --points-at HEAD | awk NF'
    ).trim()
}

String[] py_version_list = ['3.6', '3.7', '3.8']
String[] cuda_version_list = ['cpu', '92', '102', '110', '111']
String[] torch_version_list = ['1.7.0', '1.8.1']

pipeline {
    environment {
        GIT_COMMIT = getGitCommit()
        GIT_TAG = getCommitTag()
    }
    agent {
        dockerfile {
            args '--user 0:0 --gpus all'
        }
    }
    stages {
        stage('pre-install') {
            steps {
                script {
                    println "inputs:  ${env.GIT_COMMIT} - ${env.GIT_TAG} - ${env.BRANCH_NAME}"
                    if (env.BRANCH_NAME =~ /docs/ || env.GIT_COMMIT =~ /\[docs\]/) {
                        env.DOCS = 'TRUE'
                    } else {
                        env.DOCS = 'FALSE'
                    }
                    if (env.BRANCH_NAME == 'master' && (env.GIT_COMMIT =~ /\[ci\-deploy\]/ || env.GIT_TAG)) {
                        env.DEPLOY = 'TRUE'
                    } else {
                        env.DEPLOY = 'FALSE'
                    }
                    println "outputs ${env.DOCS} - ${env.DEPLOY}"
                }
            }
        }
        stage('deployment') {
            steps {
                script {
                    for (py_version in py_version_list) {
                        env.PY_VERSION = py_version
                        for (torch_version in torch_version_list) {
                            env.TORCH_VERSION = torch_version
                            for (cuda_version in cuda_version_list) {
                                env.CUDA_VERSION = cuda_version
                                env.CONDA_ENV = "${env.PY_VERSION}_${env.TORCH_VERSION}_${env.CUDA_VERSION}"
                                if ((torch_version == '1.7.0' && cuda_version == '111') ||
                                    (torch_version == '1.8.0' && cuda_version == '92')) {
                                    continue;
                                }
                                if (env.DEPLOY == 'FALSE') {
                                    if (py_version == '3.6' && (cuda_version == '110' || cuda_version == '111')) {
                                    } else {
                                        continue
                                    }
                                }
                                stage('build') {
                                    sh 'bash ./scripts/cuda.sh'
                                    sh 'bash ./scripts/conda.sh'
                                    sh 'conda install pytorch=${TORCH_VERSION} ${TOOLKIT} -c pytorch -c conda-forge --yes -n ${env.CONDA_ENV}'
                                    sh 'conda run -n ${env.CONDA_ENV} pip install --no-cache-dir --editable ./keops/'
                                    sh 'conda run -n ${env.CONDA_ENV} pip install -v --editable .[test,doc]'
                                }
                                stage('test') {
                                    sh 'conda run -n ${env.CONDA_ENV} flake8 --count falkon'
                                    sh 'conda run -n ${env.CONDA_ENV} pytest --cov-report=term-missing --cov-report=xml:coverage.xml --junitxml=junit.xml --cov=falkon --cov-config setup.cfg'
                                }
                                /*post {
                                    success {  // post test-coverage results to codecov website
                                        junit 'junit.xml'
                                        withCredentials([string(credentialsId: 'CODECOV_TOKEN', variable: 'CODECOV_TOKEN')]) {
                                            sh 'curl -s https://codecov.io/bash | bash -s -- -c -f coverage.xml -t $CODECOV_TOKEN'
                                        }
                                    }
                                }*/
                                if (env.DEPLOY == 'TRUE') {
                                    stage('deploy') {
                                        sh 'python setup.py bdist_wheel --dist-dir=dist'
                                        sh 'ls -lah dist/'
                                    }
                                }
                                if (env.DEPLOY == 'TRUE' || env.DOCS == 'TRUE') {
                                    stage('docs') {
                                        sh 'python -m pip install --upgrade --progress-bar off ghp-import'
                                        withCredentials([string(credentialsId: 'GIT_TOKEN', variable: 'GIT_TOKEN')]) {
                                            sh "/sbin/start-stop-daemon --start --quiet --pidfile /tmp/custom_xvfb_99.pid --make-pidfile --background --exec /usr/bin/Xvfb -- :99 -screen 0 1400x900x24 -ac +extension GLX +render -noreset"
                                            sh 'git remote set-url origin https://Giodiro:${GIT_TOKEN}@github.com/FalkonML/falkon.git'
                                            sh '''
                                            cd ./doc
                                            make clean && make html && make install
                                            '''
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    post {
        cleanup {
            cleanWs()
        }
    }
}
