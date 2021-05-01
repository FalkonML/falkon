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
    if (toolkit_path == "") {
        return start_path
     }
    env.CUDA_HOME = "${toolkit_path}"
    env.LD_LIBRARY_PATH = "${toolkit_path}/lib64/:${toolkit_path}/extras/CUPTI/lib64:/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
    return "${toolkit_path}/bin:${start_path}"
}

String[] py_version_list = ['3.6', '3.7', '3.8']
String[] cuda_version_list = ['cpu', '9.2', '10.2', '11.0', '11.1']
String[] torch_version_list = ['1.7.0', '1.8.1']
original_path = '/opt/conda/bin:/opt/rh/devtoolset-7/root/usr/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'

pipeline {
    agent any
    environment {
        GIT_COMMIT = getGitCommit()
        GIT_TAG = getCommitTag()
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
                    if (env.GIT_COMMIT =~ /\[ci\-deploy\]/ || env.GIT_TAG) {
                        env.DEPLOY = 'TRUE'
                    } else {
                        env.DEPLOY = 'FALSE'
                    }
                    println "outputs ${env.DOCS} - ${env.DEPLOY}"
                }
            }
        }
        stage('main-pipeline') {
            steps {
                script {
                    for (py_version in py_version_list) {
                        for (torch_version in torch_version_list) {
                            for (cuda_version in cuda_version_list) {
                                env.PY_VERSION = py_version
                                env.TORCH_VERSION = torch_version
                                env.CUDA_VERSION = cuda_version
                                env.CONDA_ENV = "PY${env.PY_VERSION}_TORCH${env.TORCH_VERSION}_CU${env.CUDA_VERSION}"
                                /* Filter out non-interesting versions. Some combos don't work, some are too long to test */
                                if ((torch_version == '1.7.0' && cuda_version == '11.1') ||
                                    (torch_version == '1.8.1' && cuda_version == '9.2')) {
                                    continue;
                                }
                                if (env.DEPLOY == 'FALSE') {
                                    if (py_version == '3.6' && (cuda_version == '9.2')) {
                                    } else {
                                        continue
                                    }
                                } else { // TODO: Temporary filters
                                    if ((py_version == '3.6' && cuda_version == '10.2') || (py_version == '3.6' && cuda_version == 'cpu')) {}
                                    else { continue }
                                }

                                def docker_tag = ''
                                if (cuda_version == 'cpu') {
                                    docker_tag = 'cpu'
                                } else {
                                    docker_tag = "cuda${cuda_version}"
                                }
                                withCredentials([string(credentialsId: 'CODECOV_TOKEN', variable: 'CODECOV_TOKEN'),
                                                 string(credentialsId: 'GIT_TOKEN', variable: 'GIT_TOKEN')]) {
                                    try {
                                        sh "CUDA_VERSION=${cuda_version} scripts/build_docker.sh"
                                        sh """
                                        docker run --rm -t \
                                            -e CUDA_VERSION=${cuda_version} \
                                            -e PYTHON_VERSION=${py_version} \
                                            -e PYTORCH_VERSION=${torch_version} \
                                            -e WHEEL_FOLDER=/falkon/dist \
                                            -e CODECOV_TOKEN=\${CODECOV_TOKEN} \
                                            -e GIT_TOKEN=\${GIT_TOKEN} \
                                            -e BUILD_DOCS=${DOCS} \
                                            -e UPLOAD_CODECOV=${DOCS} \
                                            -v \$(pwd):/falkon \
                                            --user 0:0 \
                                            --gpus all \
                                            falkon/build:${docker_tag} \
                                            scripts/build_falkon.sh
                                        """
                                    } finally {
                                        def currentResult = currentBuild.result ?: 'SUCCESS'
                                        if (currentResult == 'SUCCESS') {
                                            archiveArtifacts artifacts: "dist/**/*.whl", fingerprint: true
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
