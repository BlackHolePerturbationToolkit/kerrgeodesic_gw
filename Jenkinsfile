pipeline {
  agent {
    docker {
      image 'sage-kerrgeodesics_gw'
    }

  }
  stages {
    stage('Run tests') {
      steps {
        dir(path: 'kerrgeodesic_gw') {
          checkout scm
        }
        sh 'sage -pip install --upgrade --no-index -v -e kerrgeodesic_gw'
        dir(path: 'kerrgeodesic_gw') {
          sh 'sage -tp 4 --verbose --logfile TestReport.log kerrgeodesic_gw'
        }

      }
    }
  }
  post {
    always {
      cleanWs()
    }
  }
  options {
    skipDefaultCheckout(true)
  }
}
