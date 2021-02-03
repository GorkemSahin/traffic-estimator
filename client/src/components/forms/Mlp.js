import React from 'react';
import { Form, InputNumber, Button } from 'antd';
import MonthPicker from '../MonthPicker';

function Mlp({ setFile }) {

  return (
    <>
      <div style={{ display: 'flex' }}>
        <div style={{ display: 'flex', flexDirection: 'column', width: '300px' }}>
          <Form.Item
            label="Epoch"
            name="num_epochs"
            rules={[{ required: true }]}
          >
            <InputNumber min={1}/>
          </Form.Item>
          <Form.Item
            label="Learning rate"
            name="learning_rate"
            rules={[{ required: true }]}
          >
            <InputNumber min={0} max={1}/>
          </Form.Item>
          <Form.Item
            label="Neurons"
            name="neuron_count"
            rules={[{ required: true }]}
          >
            <InputNumber min={1} />
          </Form.Item>
          <Form.Item
            label="Dropout Rate"
            name="dropout_rate"
            rules={[{ required: true }]}
          >
            <InputNumber min={0} max={1} />
          </Form.Item>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', width: '300px' }}>
          <MonthPicker setFile={setFile}/>
        </div>
      </div>
      <Form.Item wrapperCol={{ offset: 11, span: 16 }}>
        <Button type="primary" htmlType="submit">
          Submit
        </Button>
      </Form.Item>
    </>
  );
}

export default Mlp;
