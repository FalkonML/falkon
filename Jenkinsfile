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
String[] cuda_version_list = ['cpu', '9.2', '10.2', '11.0', '11.1']
String[] torch_version_list = ['1.7.1', '1.8.1']
build_docs = false
full_deploy = false


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
                    if (env.BRANCH_NAME =~ /docs/ || env.GIT_COMMIT =~ /\[docs\]/) {
                        build_docs = true
                    } else {
                        build_docs = false
                    }
                    if (env.GIT_COMMIT =~ /\[ci\-deploy\]/ || env.GIT_TAG) {
                        full_deploy = true
                    } else {
                        full_deploy = false
                    }
                    println "Build-docs is ${build_docs} -- Full-deploy is ${full_deploy}"
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

                                def will_process = true
                                stage("filter-${CONDA_ENV}") {
                                    def reason = ""
                                    /* Filter out non-interesting versions. Some combos don't work, some are too long to test */
                                    if ((torch_version == '1.7.1' && cuda_version == '11.1') ||  // Doesn't work?
                                        (torch_version == '1.8.1' && cuda_version == '9.2') ||   // CUDA too old, not supported
                                        (torch_version == '1.8.1' && cuda_version == '11.0'))     // No point using 11.0 when 11.1 is available.
                                    {
                                        will_process = false
                                        reason = "This configuration is invalid"
                                    }
                                    if (!full_deploy) {
                                        if ((torch_version == '1.7.1' && py_version == '3.8' && cuda_version == '11.0')) {}
                                        else {
                                            will_process = false
                                            reason = "This configuration is only processed when running a full deploy"
                                        }
                                    } else { // TODO: Temporary filters
                                        if ((torch_version == '1.7.1' && py_version == '3.8' && cuda_version == '11.0')) {}
                                        else {  
                                            will_process = false
                                            reason = "This configuration has been temporarily excluded from full deploy"
                                        }
                                    }

                                    // Docs should only be built once
                                    if (build_docs && torch_version == '1.8.1' && py_version == '3.8' && cuda_version == '11.1') {
                                        env.DOCS = 'TRUE';
                                    } else {
                                        env.DOCS = 'FALSE';
                                    }
                                    if (!will_process) {
                                        unstable("${reason}")
                                    }
                                }
                                if (!will_process) {
                                    continue
                                }

                                stage("build-${env.CONDA_ENV}") {
                                    def build_success = false
                                    def docker_tag = ''
                                    if (cuda_version == 'cpu') {
                                        docker_tag = 'cpu'
                                    } else {
                                        docker_tag = "cuda${cuda_version}"
                                    }
                                    withCredentials([string(credentialsId: 'CODECOV_TOKEN', variable: 'CODECOV_TOKEN'),
                                                     string(credentialsId: 'GIT_TOKEN', variable: 'GIT_TOKEN')]) {
                                        try {
                                            // If this fails abort immediately
                                            catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                                                sh "CUDA_VERSION=${cuda_version} scripts/build_docker.sh"
                                            }
                                            // If this fails, we can keep going to the next configuration.
                                            catchError(buildResult: 'SUCCESS', stageResult: 'UNSTABLE') {
                                                sh """
                                                docker run --rm -t \
                                                    -e CUDA_VERSION=${cuda_version} \
                                                    -e PYTHON_VERSION=${py_version} \
                                                    -e PYTORCH_VERSION=${torch_version} \
                                                    -e WHEEL_FOLDER=/falkon/dist \
                                                    -e CODECOV_TOKEN=\${CODECOV_TOKEN} \
                                                    -e GIT_TOKEN=\${GIT_TOKEN} \
                                                    -e BUILD_DOCS=${env.DOCS} \
                                                    -e UPLOAD_CODECOV=${env.DOCS} \
                                                    -e HOME_DIR=\$(pwd) \
                                                    --mount type=volume,source=${env.VOLUME_NAME},destination=/jenkins_data \
                                                    --user 0:0 \
                                                    --gpus all \
                                                    falkon/build:${docker_tag} \
                                                    /falkon/scripts/build_falkon.sh   
                                                """
                                                build_success = true
                                            }
                                        } finally {
                                            if (build_success) {
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
    }
    post {
        cleanup {
            cleanWs()
        }
    }
}
