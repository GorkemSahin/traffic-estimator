import React, { useState } from 'react';
import { Layout, Typography, Radio, Form } from 'antd';
import {
  StyledHeader,
  StyledFooter,
  StyledContent,
} from './styled';
import Sarima from './components/forms/Sarima';
import Mlp from './components/forms/Mlp';
import Lstm from './components/forms/Lstm';
import Chart from './components/Chart';
import axiosInstance from './api';
import './App.less';

const { Title } = Typography;

function App() {

  const [selection, select] = useState();
  const [data, setData] = useState();
  const [file, setFile] = useState();
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState(false);

  const onSubmit = async (values) => {
    setIsTraining(true);
    const data = {
      ...values,
      file,
      start: {
        day: values.start.date(),
        month: values.start.month() + 1,
        year: values.start.year(),
      },
      test_start: {
        day: values.test_start.date(),
        month: values.test_start.month() + 1,
        year: values.test_start.year(),
      },
      end: {
        day: values.end.date(),
        month: values.end.month() + 1,
        year: values.end.year(),
      }
    }
    try {
      const resp = await axiosInstance.post(selection, data);
      setData(resp.data);
    } catch (e) {
      setError(true)
    } finally {
      setIsTraining(false)
    }
  }

  return (
    <Layout>
      <StyledHeader>
        <Title style={{ color: '#1db954'}}>Traffic Estimator</Title>
      </StyledHeader>
      <StyledContent>
        { data
          ? <Chart data={ data }/>
          : error
          ? <Title style={{ color: '#1db954'}} level={3}>An error occured</Title>
          : isTraining
          ? <Title style={{ color: '#1db954'}} level={3}>Training...</Title>
          : (<>
              <div style={{ display: 'flex', height: '50px', alignItems: 'center', justifyContent: 'center' }}>
                <Radio.Group style={{ color: '#1db954' }} onChange={ (e) => select(e.target.value) } value={ selection }>
                  <Radio value={ '/sarima' }>SARIMA</Radio>
                  <Radio value={ '/mlp' }>MLP</Radio>
                  <Radio value={ '/lstm' }>LSTM</Radio>
                </Radio.Group>
              </div>
              <div style={{ display: 'flex', height: '400px' }}>
                <Form
                  onFinish={ onSubmit }
                  labelCol={{ span: 15 }}
                  layout="horizontal"
                >
                  { selection === '/sarima' &&
                    <Sarima setFile={setFile}/>
                  }
                  { selection === '/mlp' &&
                    <Mlp setFile={setFile}/>
                  }
                  { selection === '/lstm' &&
                    <Lstm setFile={setFile}/>
                  }
                </Form>
              </div>
          </>)
        }
      </StyledContent>
      <StyledFooter>Görkem Şahin • Yağız Akyüz</StyledFooter>
    </Layout>
  );
}

export default App;
