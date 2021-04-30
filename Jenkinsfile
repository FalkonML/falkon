def getGitCommit() {
    return sh(script: "git log -1 --pretty=%B", returnStdout: true)
}

def getCommitTag() {
    return sh(
        returnStdout: true,
        script: 'git fetch --tags && git tag --points-at HEAD | awk NF'
    ).trim()
}

def getToolkitPackage(cuda_version) {
    if (cuda_version == 'cpu') {
        return 'cpuonly'
    } else if (cuda_version == '92') {
        return 'cudatoolkit=9.2'
    } else if (cuda_version == '102') {
        return 'cudatoolkit=10.2'
    } else if (cuda_version == '110') {
        return 'cudatoolkit=11.0'
    } else if (cuda_version == '111') {
        return 'cudatoolkit=11.1'
    }
    return ''
}

def setupCuda(start_path) {
    def toolkit_path = sh(
        returnStdout: true,
        script: 'bash ./scripts/cuda.sh'
    ).trim()
    env.CUDA_HOME = "${toolkit_path}"
    env.LD_LIBRARY_PATH = "${toolkit_path}/lib64/:${toolkit_path}/extras/CUPTI/lib64:/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
    return "${toolkit_path}/bin:${start_path}"
}

String[] py_version_list = ['3.6', '3.7', '3.8']
String[] cuda_version_list = ['cpu', '92', '102', '110', '111']
String[] torch_version_list = ['1.7.0', '1.8.1']
original_path = '/opt/conda/bin:/opt/rh/devtoolset-7/root/usr/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'

pipeline {
    environment {
        GIT_COMMIT = getGitCommit()
        GIT_TAG = getCommitTag()
    }
    agent {
        dockerfile {
            args '--gpus all --user 0:0'
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
                                env.CONDA_ENV = "PY${env.PY_VERSION}_TORCH${env.TORCH_VERSION}_CU${env.CUDA_VERSION}"
                                if ((torch_version == '1.7.0' && cuda_version == '111') ||
                                    (torch_version == '1.8.0' && cuda_version == '92')) {
                                    continue;
                                }
                                if (env.DEPLOY == 'FALSE') {
                                    if (py_version == '3.6' && (cuda_version == '92')) { //(cuda_version == '110' || cuda_version == '111')) {
                                    } else {
                                        continue
                                    }
                                }
                                /* BUILD */
                                def toolkit = getToolkitPackage(cuda_version)
                                sh 'bash ./scripts/conda.sh'
                                def new_path = setupCuda(original_path)

                                stage("build-${env.CONDA_ENV}") {
                                    // We need this trick since otherwise it's impossible to modify PATH!
                                    sh """
                                    export PATH=${new_path}
                                    conda install -q pytorch=${env.TORCH_VERSION} ${toolkit} -c pytorch -c conda-forge --yes -n ${env.CONDA_ENV}
                                    conda run -n ${env.CONDA_ENV} pip install --no-cache-dir --editable ./keops/
                                    conda run -n ${env.CONDA_ENV} pip install -v --editable .[test,doc]
                                    """
                                }
                                /* TESTING */
                                try {
                                    stage("test-${env.CONDA_ENV}") {
                                        sh "PATH=${new_path} conda run -n ${env.CONDA_ENV} pytest 'falkon/tests/test_kernels.py::TestLaplacianKernel::test_mmv[No KeOps-gpu]'"
                                        sh "conda run -n ${env.CONDA_ENV} flake8 --count falkon"
                                        sh "PATH=${new_path} conda run -n ${env.CONDA_ENV} pytest --cov-report=term-missing --cov-report=xml:coverage.xml --junitxml=junit.xml --cov=falkon --cov-config setup.cfg"
                                    }
                                } finally {
                                    def currentResult = currentBuild.result ?: 'SUCCESS'
                                    if (currentResult == 'SUCCESS') {
                                        // post test-coverage results to codecov website
                                        junit 'junit.xml'
                                        withCredentials([string(credentialsId: 'CODECOV_TOKEN', variable: 'CODECOV_TOKEN')]) {
                                            sh 'curl -s https://codecov.io/bash | bash -s -- -c -f coverage.xml -t $CODECOV_TOKEN'
                                        }
                                    }
                                }
                                /* DEPLOYMENT */
                                if (env.DEPLOY == 'TRUE') {
                                    def dist_dir = "torch-${env.TORCH_VERSION}+${env.CUDA_VERSION}"
                                    try {
                                        stage("deploy-${env.CONDA_ENV}") {
                                            sh """
                                            export PATH=${new_path}
                                            mkdir ./dist
                                            conda run -n ${env.CONDA_ENV} python setup.py bdist_wheel --dist-dir=./dist/${dist_dir}
                                            ls -lah ./dist/${dist_dir}
                                            """
                                        }
                                    } finally {
                                        def currentResult = currentBuild.result ?: 'SUCCESS'
                                        if (currentResult == 'SUCCESS') {
                                            archiveArtifacts artifacts: "dist/**/*.whl", fingerprint: true
                                        }
                                    }
                                }
                                /* DOCUMENTATION DEPLOYMENT */
                                if (env.DOCS == 'TRUE') {
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
