import React from 'react';
import { Form, InputNumber, Button } from 'antd';
import MonthPicker from '../MonthPicker';

function Sarima({ setFile }) {

  return (
    <>
      <div style={{ display: 'flex' }}>
        <div style={{ display: 'flex', flexDirection: 'column', width: '300px' }}>
          <Form.Item
            label="Autoregression"
            name="p"
            rules={[{ required: true }]}
          >
            <InputNumber min={0} max={10}/>
          </Form.Item>
          <Form.Item
            label="Integration"
            name="d"
            rules={[{ required: true }]}
          >
            <InputNumber min={0} max={10}/>
          </Form.Item>
          <Form.Item
            label="Moving average"
            name="q"
            rules={[{ required: true }]}
          >
            <InputNumber min={0} max={10}/>
          </Form.Item>
          <Form.Item
            label="Seasonal autoregression"
            name="P"
            rules={[{ required: true }]}
          >
            <InputNumber min={0} max={10}/>
          </Form.Item>
          <Form.Item
            label="Seasonal integration"
            name="D"
            rules={[{ required: true }]}
          >
            <InputNumber min={0} max={10}/>
          </Form.Item>
          <Form.Item
            label="Seasonal moving average"
            name="Q"
            rules={[{ required: true }]}
          >
            <InputNumber min={0} max={10}/>
          </Form.Item>
          <Form.Item
            label="Seasonality"
            name="s"
            rules={[{ required: true }]}
          >
            <InputNumber min={0} max={1000}/>
          </Form.Item>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', width: '300px' }}>
          <MonthPicker setFile={setFile}/>
        </div>
      </div>
      <Form.Item wrapperCol={{ offset: 11 }}>
        <Button type="primary" htmlType="submit">
          Submit
        </Button>
      </Form.Item>
    </>
  );
}

export default Sarima;
