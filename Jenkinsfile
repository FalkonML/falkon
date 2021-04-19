def getGitCommit() {
    return sh(script: "git log -1 --pretty=%B", returnStdout: true)
}

def commitHasTag() {
    if (env.BRANCH_NAME == 'master') {
        TAG = sh (
            returnStdout: true,
            script: 'git fetch --tags && git tag --points-at HEAD | awk NF'
        ).trim()
        if (TAG) {
            return true
        }
        return false
    }
    return false
}

pipeline {
    environment {
        GIT_COMMIT = getGitCommit()
        HAS_TAG = commitHasTag()
        DOCS = ''
        DEPLOY = ''
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
                    println env.GIT_COMMIT
                    println env.HAS_TAG
                    def do_docs = env.GIT_BRANCH == 'docs' || env.GIT_COMMIT =~ '.*/[docs/].*'
                    env.DOCS = do_docs ? 'TRUE' : 'FALSE'
                    def do_deploy = env.GIT_COMMIT =~ '.*/[ci-deploy/].*' || env.HAS_TAG
                    env.DEPLOY = do_deploy ? 'TRUE' : 'FALSE'
                }
            }
        }
        stage('build-test') {
            matrix {
                axes {
                    axis {
                        name 'PY_VERSION'
                        values '3.6', '3.7', '3.8'
                    }
                    axis {
                        name 'TORCH_VERSION'
                        values '1.7.0', '1.8.1'
                    }
                    axis {
                        name 'CUDA_VERSION'
                        values 'cpu', '92', '102', '110', '111'
                    }
                }
                excludes {
                    exclude {
                        axis {
                            name 'TORCH_VERSION'
                            values '1.7.0'
                        }
                        axis {
                            name 'CUDA_VERSION'
                            values '111'
                        }
                    }
                    exclude {
                        axis {
                            name 'TORCH_VERSION'
                            values '1.8.0'
                        }
                        axis {
                            name 'CUDA_VERSION'
                            values '92'
                        }
                    }
                }
                when {
                    // DEPLOY is True => all cells are fine
                    // DEPLOY is False => only Py3.6, 11.0/11.1
                    anyOf {
                        allOf {
                            environment name: 'DEPLOY', value: 'TRUE'
                        }
                        allOf {
                            environment name: 'DEPLOY', value: 'FALSE'
                            environment name: 'PY_VERSION', value: '3.6'
                            anyOf {
                                environment name: 'CUDA_VERSION', value: '110'
                                environment name: 'CUDA_VERSION', value: '111'
                            }
                        }
                    }
                }
                stages {
                    stage('build') {
                        steps {
                            sh 'scripts/cuda.sh'
                            sh 'scripts/conda.sh'
                            sh 'conda install pytorch=${TORCH_VERSION} ${TOOLKIT} -c pytorch -c conda-forge --yes'
                            sh 'pip install --no-cache-dir --editable ./keops/'
                            sh 'pip install -v --editable .[test,doc]'
                        }
                    }
                    stage('test') {
                        steps {
                            sh 'flake8 --count falkon'
                            sh 'pytest --cov-report=term-missing --cov-report=xml:coverage.xml --junitxml=junit.xml --cov=falkon --cov-config setup.cfg'
                        }
                        post {
                            success {  // post test-coverage results to codecov website
                                junit 'junit.xml'
                                withCredentials([string(credentialsId: 'CODECOV_TOKEN', variable: 'CODECOV_TOKEN')]) {
                                    sh 'curl -s https://codecov.io/bash | bash -s -- -c -f coverage.xml -t $CODECOV_TOKEN'
                                }
                            }
                        }
                    }
                    stage('deploy') {
                        when {
                            environment name: 'DEPLOY', value: 'TRUE'
                        }
                        steps {
                            sh 'python setup.py bdist_wheel --dist-dir=dist'
                            sh 'ls -lah dist/'
                        }
//                         post {
//                             success {
//
//                             }
//                         }
                    }
                    stage('docs') {
                        when {
                            anyOf {
                                environment name: 'DEPLOY', value: 'TRUE'
                                environment name: 'DOCS', value: 'TRUE'
                            }
                        }
                        steps {
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
    post {
        cleanup {
            cleanWs()
        }
    }
}
