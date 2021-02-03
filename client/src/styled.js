import styled from 'styled-components';
import { Layout } from 'antd';
const { Header, Footer, Content } = Layout;

export const StyledHeader = styled(Header)`
  background-color: #212121;
  padding: 1em;
`;

export const StyledFooter = styled(Footer)`
  background-color: #212121;
  text-align: center;
`;

export const StyledContent = styled(Content)`
  display: flex;
  flex: 1;
  flex-direction: column;
  background-color: #212121;
  align-items: center;
  justify-content: center;
  overflow: hidden;
`;