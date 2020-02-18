#!/bin/bash -e

mkdir data

wget http://ixa2.si.ehu.es/stswiki/images/e/e4/STS2012-en-train.zip
tar zxvf STS2012-en-train.zip
mv train data/2012TRAIN

wget http://ixa2.si.ehu.es/stswiki/images/4/40/STS2012-en-test.zip
tar zxvf STS2012-en-test.zip
mv test-gold data/2012GOLD

wget http://ixa2.si.ehu.es/stswiki/images/2/2f/STS2013-en-test.zip
tar zxvf STS2013-en-test.zip
mv test-gs data/2013GOLD

wget http://ixa2.si.ehu.es/stswiki/images/8/8c/STS2014-en-test.zip
tar zxvf STS2014-en-test.zip
mv sts-en-test-gs-2014 data/2014GOLD

wget http://ixa2.si.ehu.es/stswiki/images/d/da/STS2015-en-test.zip
tar zxvf STS2015-en-test.zip
mv test_evaluation_task2a data/2015GOLD

rm STS2012-en-train.zip STS2012-en-test.zip STS2013-en-test.zip STS2014-en-test.zip STS2015-en-test.zip
