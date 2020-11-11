pipeline {
    agent {
        dockerfile {
            args '--user 0:0'
        }
    }
    stages {
        boolean test_passed = true
        boolean flake8_passed = true
        stage('build') {
            when {
                expression {
                    GIT_BRANCH = 'origin/' + sh(returnStdout: true, script: 'git rev-parse --abbrev-ref HEAD').trim()
                    return !(GIT_BRANCH ==~ /(?i)\[skip ci\]/)
                }
            }
            steps {
                sh 'pip install --no-cache-dir --editable ./keops/'
                // Need editable in order for docs to build correctly
                sh 'pip install --no-cache-dir -v --editable .[test,doc]'
            }
        }
        stage('test') {
            when {
                expression {
                    GIT_BRANCH = 'origin/' + sh(returnStdout: true, script: 'git rev-parse --abbrev-ref HEAD').trim()
                    return !(GIT_BRANCH ==~ /(?i)\[skip ci\]/)
                }
            }
            steps {
                try {
                    sh 'pytest --cov-report=term-missing --cov-report=xml:coverage.xml --junitxml=junit.xml --cov=falkon --cov-config setup.cfg'
                } catch (Exception e) {
                    test_passed = false
                    error "Tests failed"
                }
                try {
                    sh 'flake8 --count falkon'
                } catch (Exception e) {
                    flake8_passed = false
                    error "Flake8 failed"
                }
            }
            post {
                always {
                    junit 'junit.xml'
                    withCredentials([string(credentialsId: 'CODECOV_TOKEN', variable: 'CODECOV_TOKEN')]) {
                        sh 'curl -s https://codecov.io/bash | bash -s -- -c -f coverage.xml -t $CODECOV_TOKEN'
                    }
                }
            }
        }
        stage('build-docs') {
            when {
                expression {
                    GIT_BRANCH = 'origin/' + sh(returnStdout: true, script: 'git rev-parse --abbrev-ref HEAD').trim()
                    return !(GIT_BRANCH ==~ /(?i)\[skip ci\]/)
                }
            }
            steps {
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
    post {
        cleanup {
            cleanWs()
        }
    }
}